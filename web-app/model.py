import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from sklearn.metrics import r2_score


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, bias=True,
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, strides, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class Local_Seq_Conv(nn.Module):
    def __init__(self, input_channel, seq_len, kernel_size, output_channels=64):
        super(Local_Seq_Conv, self).__init__()
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.strides = 1

        self.conv2d = Conv2dSame(in_channels=input_channel, out_channels=output_channels,
                                 kernel_size=kernel_size, strides=1)

    def forward(self, x):
        output = []
        for i in range(self.seq_len):
            tmp = self.conv2d(x[:, i, :, :, :])
            tmp = torch.relu(tmp)
            output.append(tmp)
        output = torch.stack(output, dim=1)
        return output


class DMVSTNet_Landsat(nn.Module):
    def __init__(self, seq_len, input_channels, height, width, conv_len=3,
                 cnn_hidden_dim=64, kernel_size=3, hidden_dim=128, spatial_out_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.height = height
        self.width = width
        self.cnn_hidden_dim = cnn_hidden_dim
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        # Spatial view: Local CNN layers
        convs = []
        for i in range(conv_len):
            if i == 0:
                convs.append(Local_Seq_Conv(input_channels, seq_len, kernel_size, cnn_hidden_dim))
            else:
                convs.append(Local_Seq_Conv(cnn_hidden_dim, seq_len, kernel_size, cnn_hidden_dim))
        self.local_cnn = nn.Sequential(*convs)

        # After CNN: flatten and reduce dimensions
        self.flatten = nn.Flatten(start_dim=2)  # Keep batch and seq dims

        # Calculate flattened size after conv layers
        # After conv layers with same padding, spatial dims remain unchanged
        cnn_output_size = cnn_hidden_dim * height * width

        self.spatial_dense = nn.Linear(cnn_output_size, spatial_out_dim)

        # Temporal view: LSTM
        self.lstm = nn.LSTM(input_size=spatial_out_dim, hidden_size=hidden_dim, batch_first=False)

        # Final prediction layer
        self.output = nn.Linear(in_features=hidden_dim, out_features=height * width)

    def forward(self, x):
        # Fix: [batch, seq_len, height, width, channels] -> [batch, seq_len, channels, height, width]
        x = x.permute(0, 1, 4, 2, 3)
        t = self.local_cnn(x)  # [batch_size, seq_len, cnn_hidden_dim, height, width]

        t = self.flatten(t)  # [batch_size, seq_len, cnn_hidden_dim*height*width]

        spatial_out = self.spatial_dense(t)  # [batch_size, seq_len, spatial_out_dim]
        spatial_out = torch.transpose(spatial_out, 0, 1)  # [seq_len, batch_size, spatial_out_dim]

        _, (hid, cell) = self.lstm(spatial_out)
        lstm_out = hid.squeeze(0)  # [batch_size, hidden_dim]

        out = self.output(lstm_out)  # [batch_size, height*width]
        out = out.view(-1, 1, self.height, self.width, 1)  # [batch_size, 1, height, width, output_length]

        return out


class LandsatLSTPredictor(pl.LightningModule):
    def __init__(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            warmup_steps: int = 1000,
            max_epochs: int = 100,
            log_images_every_n_epochs: int = 1,
            max_images_to_log: int = 4,
            input_sequence_length: int = 3,
            output_sequence_length: int = 3,
            model_size: str = "small",  # NEW: "tiny", "small", "medium", "large", "earthnet", "lstm", "earthnet_w_index"
            input_channels: int = 9,
            **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store sequence lengths
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length

        # Image logging parameters
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.max_images_to_log = max_images_to_log

        # For Scatter Plot
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        if model_size == "lstm":
            self.model = DMVSTNet_Landsat(
                seq_len=input_sequence_length,
                input_channels=input_channels,
                height=128,
                width=128
            )
        else:
            # Model size configurations
            model_configs = {
                "tiny": {
                    'base_units': 64,
                    'num_heads': 4,
                    'enc_depth': [1, 1],
                    'dec_depth': [1, 1],
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                    'num_global_vectors': 4,
                },
                "small": {
                    'base_units': 96,
                    'num_heads': 6,
                    'enc_depth': [2, 2],
                    'dec_depth': [1, 1],
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                    'num_global_vectors': 8,
                },
                "medium": {
                    'base_units': 128,  # Keep same
                    'num_heads': 8,  # Keep same
                    'enc_depth': [2, 2],  # REDUCED from [3, 3]
                    'dec_depth': [1, 1],  # REDUCED from [2, 2]
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],  # REDUCED from [(2, 4, 4), (2, 8, 8)]
                    'num_global_vectors': 12,  # REDUCED from 16
                },
                "large": {
                    'base_units': 144,  # REDUCED from 192
                    'num_heads': 8,  # REDUCED from 12 (must divide base_units)
                    'enc_depth': [2, 2],  # REDUCED from [4, 4]
                    'dec_depth': [1, 1],  # REDUCED from [3, 3]
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],  # REDUCED from [(2, 8, 8), (2, 8, 8)]
                    'num_global_vectors': 16,  # REDUCED from 32
                },
                "earthnet": {
                    'base_units': 256,
                    'num_heads': 4,
                    'enc_depth': [1, 1],
                    'dec_depth': [1, 1],
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                    'num_global_vectors': 8,
                    'use_dec_self_global': False,  # Earthnet disables this
                    'use_dec_cross_global': False,  # Earthnet disables this
                    'initial_downsample_type': 'stack_conv',
                    'initial_downsample_stack_conv_num_layers': 2,
                    'initial_downsample_stack_conv_dim_list': [64, 256],
                    'initial_downsample_stack_conv_downscale_list': [2, 2],
                    'initial_downsample_stack_conv_num_conv_list': [2, 2],
                    'initial_downsample_activation': 'leaky',
                },
                "earthnet_w_index": {
                    'base_units': 256,
                    'num_heads': 4,
                    'enc_depth': [1, 1],
                    'dec_depth': [1, 1],
                    'enc_cuboid_size': [(2, 4, 4), (2, 4, 4)],
                    'num_global_vectors': 8,
                    'use_dec_self_global': False,  # Earthnet disables this
                    'use_dec_cross_global': False,  # Earthnet disables this
                    'initial_downsample_type': 'stack_conv',
                    'initial_downsample_stack_conv_num_layers': 2,
                    'initial_downsample_stack_conv_dim_list': [64, 256],
                    'initial_downsample_stack_conv_downscale_list': [2, 2],
                    'initial_downsample_stack_conv_num_conv_list': [2, 2],
                    'initial_downsample_activation': 'leaky',
                }
            }
            print("You are using: ", model_size)

            # Get base config for selected model size
            selected_config = model_configs[model_size]

            # Default Landsat-optimized config (shared across all sizes)
            self.model_config = {
                'input_shape': (input_sequence_length, 128, 128, input_channels),
                'target_shape': (output_sequence_length, 128, 128, 1),
                'attn_drop': 0.1,
                'proj_drop': 0.1,
                'ffn_drop': 0.1,
                'use_dec_self_global': True,
                'use_dec_cross_global': True,
                'pos_embed_type': 't+hw',
                'use_relative_pos': True,
                'ffn_activation': 'gelu',
                'enc_cuboid_strategy': [('l', 'l', 'l'), ('d', 'd', 'd')],
                'dec_cross_cuboid_hw': [(4, 4), (4, 4)],
                'dec_cross_n_temporal': [1, 2],
            }

            # Update with size-specific config
            self.model_config.update(selected_config)

            # Update with any provided kwargs (allows override)
            self.model_config.update(model_kwargs)

            # Initialize model
            from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
            self.model = CuboidTransformerModel(**self.model_config)

        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Heat index loss flag for earthnet_w_index model
        self.use_heat_index_loss = (model_size == "earthnet_w_index")
        self.model_size = model_size
        self.num_bins = 25

        # Band names for visualization
        self.band_names = ['DEM (+10k offset)', 'LST (°F)', 'Red (×10k)', 'Green (×10k)', 'Blue (×10k)',
                           'NDVI (×10k)', 'NDWI (×10k)', 'NDBI (×10k)', 'Albedo (×10k)']

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model '{model_size}' initialized with {param_count:,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def create_correlation_plot(self, all_predictions, all_targets, split_name, epoch, already_fahrenheit=False):
        try:
            # Convert and flatten
            pred_flat = torch.cat(all_predictions).detach().cpu().numpy().flatten()
            true_flat = torch.cat(all_targets).detach().cpu().numpy().flatten()

            # Remove NODATA values (0 in normalized space, or -189 in Fahrenheit)
            if already_fahrenheit:
                nodata_val = -189.0
                valid_mask = (true_flat > nodata_val + 1) & (pred_flat > nodata_val + 1)
            else:
                valid_mask = (true_flat != 0) & (pred_flat != 0)
            pred_clean = pred_flat[valid_mask]
            true_clean = true_flat[valid_mask]

            if len(pred_clean) < 100:  # Need minimum points for meaningful plot
                return None

            # Convert to Fahrenheit for interpretation (skip if already Fahrenheit)
            if already_fahrenheit:
                pred_fahrenheit = pred_clean
                true_fahrenheit = true_clean
            else:
                pred_fahrenheit = pred_clean * (211.0 - (-189.0)) + (-189.0)
                true_fahrenheit = true_clean * (211.0 - (-189.0)) + (-189.0)

            # Calculate metrics
            correlation = np.corrcoef(true_fahrenheit, pred_fahrenheit)[0, 1]
            r2 = r2_score(true_fahrenheit, pred_fahrenheit)
            mae = np.mean(np.abs(true_fahrenheit - pred_fahrenheit))
            rmse = np.sqrt(np.mean((true_fahrenheit - pred_fahrenheit) ** 2))

            # Create plot
            fig, ax = plt.subplots(figsize=(8, 8))

            # Scatter plot with transparency
            ax.scatter(true_fahrenheit, pred_fahrenheit, alpha=0.6, s=8, color='gray', edgecolors='none')

            # Perfect prediction line (diagonal)
            min_temp = min(true_fahrenheit.min(), pred_fahrenheit.min())
            max_temp = max(true_fahrenheit.max(), pred_fahrenheit.max())
            ax.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', linewidth=2, label='Perfect Prediction')

            # Labels and title
            ax.set_xlabel('Ground Truth Mean (Background Temperature) (F)', fontsize=12)
            ax.set_ylabel('Mean (Background Temperature) (F)', fontsize=12)
            ax.set_title(f'{split_name.title()} - Epoch {epoch}\n'
                         f'Correlation: {correlation:.3f}, R²: {r2:.3f}, MAE: {mae:.1f}°F, RMSE: {rmse:.1f}°F',
                         fontsize=12)

            # Equal aspect ratio and clean appearance
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Error creating correlation plot: {e}")
            return None

    def masked_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = (targets != 0).float()  # if ground_truth has NODATA>50% -> dont include, reduces sequences
        loss_elements = self.criterion(predictions, targets)
        masked_loss = loss_elements * valid_mask
        valid_count = valid_mask.sum()

        if valid_count > 0:
            return masked_loss.sum() / valid_count
        else:
            # Return a small loss that won't cause NaN gradients
            return torch.tensor(1e-8, device=predictions.device, dtype=predictions.dtype)

    def masked_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate MAE with NODATA masking"""
        valid_mask = (targets != 0).float()
        mae_elements = torch.abs(predictions - targets)
        masked_mae = mae_elements * valid_mask
        valid_count = valid_mask.sum()

        if valid_count > 0:
            return masked_mae.sum() / valid_count
        else:
            return torch.tensor(0.0, device=predictions.device)

    def heat_index_loss(self, inputs: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Heat Index Loss: Encourages predictions to follow the temperature distribution
        of the input sequence + target.
        
        Uses 5th and 95th percentiles instead of min/max to exclude anomalies.
        Bins require at least 5% occurrence in the total pixel data.
        
        For each sample:
        1. Extract LST from input sequence (channel index 1: DEM=0, LST=1, ...)
        2. Get 5th/95th percentile temperature across input LST + target LST
        3. Divide range into 25 bins
        4. Values below 5th percentile -> bin 1, above 95th percentile -> bin 25
        5. Calculate loss based on bin assignment accuracy
        """
        batch_size = inputs.shape[0]
        device = predictions.device
        
        # Extract LST channel from inputs (channel 1 after DEM)
        # inputs shape: [B, T, H, W, C] -> LST is at index 1
        input_lst = inputs[:, :, :, :, 1:2]  # [B, T_in, H, W, 1]
        
        # Combine input LST sequence with target for percentile calculation
        all_temps = torch.cat([input_lst, targets], dim=1)  # [B, T_total, H, W, 1]
        
        # Calculate per-sample percentiles (excluding nodata which is 0)
        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(batch_size):
            sample_temps = all_temps[b]  # [T_total, H, W, 1]
            sample_pred = predictions[b]  # [T_out, H, W, 1]
            sample_target = targets[b]  # [T_out, H, W, 1]
            
            # Get valid (non-zero) temperature values
            valid_mask_temps = sample_temps != 0
            valid_mask_target = sample_target != 0
            
            if valid_mask_temps.sum() < 100 or valid_mask_target.sum() < 10:
                continue
            
            valid_temps = sample_temps[valid_mask_temps]
            
            # Calculate 5th and 95th percentiles instead of min/max
            # This ensures at least 5% of pixels define each boundary
            temp_p5 = torch.quantile(valid_temps, 0.02)
            temp_p95 = torch.quantile(valid_temps, 0.98)
            
            # Avoid division by zero
            temp_range = temp_p95 - temp_p5
            if temp_range < 1e-6:
                temp_range = torch.tensor(1.0, device=device)
            
            # Convert predictions and targets to bin indices (1-25)
            # Values at or below p5 -> 0, at or above p95 -> 1
            pred_normalized = (sample_pred - temp_p5) / temp_range
            target_normalized = (sample_target - temp_p5) / temp_range
            
            # Clamp to [0, 1] then scale to [1, 25]
            # Below p5 -> bin 1, above p95 -> bin 25
            pred_bins = torch.clamp(pred_normalized, 0.0, 1.0) * (self.num_bins - 1) + 1
            target_bins = torch.clamp(target_normalized, 0.0, 1.0) * (self.num_bins - 1) + 1
            
            # Normalize bins back to [0, 1] for MSE loss
            # This keeps loss scale consistent with normalized temperature space
            pred_bins_norm = (pred_bins - 1) / (self.num_bins - 1)
            target_bins_norm = (target_bins - 1) / (self.num_bins - 1)
            
            # Calculate bin-based MSE loss on normalized bins
            bin_mse = self.criterion(pred_bins_norm, target_bins_norm)
            masked_bin_mse = bin_mse * valid_mask_target.float()
            
            if valid_mask_target.sum() > 0:
                sample_bin_loss = masked_bin_mse.sum() / valid_mask_target.sum()
                
                # Also add regular MSE for gradient flow
                temp_mse = self.criterion(sample_pred, sample_target)
                masked_temp_mse = temp_mse * valid_mask_target.float()
                sample_temp_loss = masked_temp_mse.sum() / valid_mask_target.sum()
                
                # Combine: bin loss for distribution, temp loss for precision
                total_loss = total_loss + 0.3 * sample_temp_loss + 0.7 * sample_bin_loss
                valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(1e-8, device=device, dtype=predictions.dtype)

    def extract_batch_metadata(self, batch_info: Any, batch_idx: int) -> Dict[str, Any]:
        """
        Extract metadata from the current batch.
        This method should be called during the training/validation step.
        """
        # Get the dataset to extract metadata
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
            dataset = self.trainer.datamodule.train_dataset

            # Calculate actual sample indices from batch
            batch_size = self.trainer.datamodule.batch_size
            start_idx = batch_idx * batch_size

            metadata = {
                'batch_idx': batch_idx,
                'epoch': self.current_epoch,
                'batch_size': batch_size,
                'start_sample_idx': start_idx,
                'samples_metadata': []
            }

            # Extract metadata for each sample in the batch
            for i in range(min(batch_size, self.max_images_to_log)):
                sample_idx = start_idx + i
                if sample_idx < len(dataset):
                    # Get the tile sequence info from dataset
                    if hasattr(dataset, 'tile_sequences') and sample_idx < len(dataset.tile_sequences):
                        city, tile_row, tile_col, input_months, output_months = dataset.tile_sequences[sample_idx]

                        sample_metadata = {
                            'sample_idx': sample_idx,
                            'city': city,
                            'tile_position': f"row_{tile_row:03d}_col_{tile_col:03d}",
                            'tile_row': tile_row,
                            'tile_col': tile_col,
                            'input_months': input_months,
                            'output_months': output_months,
                            'input_date_range': f"{input_months[0]} to {input_months[-1]}",
                            'output_date_range': f"{output_months[0]} to {output_months[-1]}",
                            'sequence_length': len(input_months),
                            'file_paths': self._get_file_paths(city, tile_row, tile_col, input_months + output_months)
                        }
                        metadata['samples_metadata'].append(sample_metadata)

            return metadata

        # Fallback metadata if dataset info not available
        return {
            'batch_idx': batch_idx,
            'epoch': self.current_epoch,
            'note': 'Limited metadata - dataset info not accessible'
        }

    def _get_file_paths(self, city: str, tile_row: int, tile_col: int, months: List[str]) -> Dict[str, List[str]]:
        """Get file paths for the tiles used in this sequence"""
        if not hasattr(self.trainer, 'datamodule'):
            return {}

        dataset_root = Path(self.trainer.datamodule.dataset_root)

        file_paths = {
            'dem_path': str(dataset_root / "DEM_2014_Tiles" / city / f"DEM_row_{tile_row:03d}_col_{tile_col:03d}.tif"),
            'monthly_scenes': {}
        }

        # Get paths for each month's tiles
        for month in months:
            monthly_scenes = self._get_monthly_scenes_for_city(city)
            if month in monthly_scenes:
                scene_dir = Path(monthly_scenes[month])
                month_paths = {}

                band_names = ['LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
                for band in band_names:
                    tile_path = scene_dir / f"{band}_row_{tile_row:03d}_col_{tile_col:03d}.tif"
                    month_paths[band] = str(tile_path)

                file_paths['monthly_scenes'][month] = month_paths

        return file_paths

    def _get_monthly_scenes_for_city(self, city: str) -> Dict[str, str]:
        """Helper to get monthly scenes for a city (simplified version of dataset method)"""
        if not hasattr(self.trainer, 'datamodule'):
            return {}

        dataset_root = Path(self.trainer.datamodule.dataset_root)
        city_dir = dataset_root / "Cities_Tiles" / city

        if not city_dir.exists():
            return {}

        monthly_scenes = {}
        scene_dirs = [d for d in city_dir.iterdir() if d.is_dir()]

        for scene_dir in scene_dirs:
            try:
                from datetime import datetime
                date_str = scene_dir.name
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                month_key = f"{date_obj.year}-{date_obj.month:02d}"

                if month_key not in monthly_scenes:
                    monthly_scenes[month_key] = str(scene_dir)
            except:
                continue

        return monthly_scenes

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with Fahrenheit temperature metrics"""
        inputs, targets = batch

        # Forward pass
        predictions = self.forward(inputs)

        # Calculate loss (use heat_index_loss for earthnet_w_index model)
        if self.use_heat_index_loss:
            loss = self.heat_index_loss(inputs, predictions, targets)
        else:
            loss = self.masked_loss(predictions, targets)

        # Calculate metrics
        mae = self.masked_mae(predictions, targets)

        # Log metrics
        self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate temperature-specific metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            temp_mae_f = self.masked_mae(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(self.masked_loss(pred_fahrenheit, true_fahrenheit))

            self.log('train_mae_F', temp_mae_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_rmse_F', temp_rmse_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store predictions for correlation plot (limit to avoid memory issues)
        if self.trainer.is_global_zero and len(self.train_predictions) < 50:  # Limit to ~50 batches per epoch
            self.train_predictions.append(predictions.detach().cpu())
            self.train_targets.append(targets.detach().cpu())

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step with direct image logging that works in sweeps"""
        inputs, targets = batch
        predictions = self.forward(inputs)

        # Calculate loss in normalized space (use heat_index_loss for earthnet_w_index model)
        if self.use_heat_index_loss:
            loss = self.heat_index_loss(inputs, predictions, targets)
        else:
            loss = self.masked_loss(predictions, targets)
        mae = self.masked_mae(predictions, targets)

        # Log normalized metrics
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_mae', mae, on_step=False, on_epoch=True, sync_dist=True)

        # Calculate metrics in Fahrenheit
        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)

            temp_mae_f = self.masked_mae(pred_fahrenheit, true_fahrenheit)
            temp_rmse_f = torch.sqrt(self.masked_loss(pred_fahrenheit, true_fahrenheit))

            # Log ACTUAL temperature metrics
            self.log('val_mae_F', temp_mae_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_rmse_F', temp_rmse_f, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

            # Correlation (same in both spaces)
            pred_flat = pred_fahrenheit.flatten()
            true_flat = true_fahrenheit.flatten()
            # Add NODATA masking to correlation
            valid_data_mask = (true_flat != 0) & (pred_flat != 0)  # Exclude NODATA
            finite_mask = torch.isfinite(pred_flat) & torch.isfinite(true_flat)
            mask = valid_data_mask & finite_mask
            if mask.sum() > 1:
                correlation = torch.corrcoef(torch.stack([pred_flat[mask], true_flat[mask]]))[0, 1]
                if torch.isfinite(correlation):
                    self.log('val_correlation', correlation, on_step=False, on_epoch=True, sync_dist=True)

        # Store predictions for epoch-end metrics (store in Fahrenheit like test)
        self.val_predictions.append(pred_fahrenheit.detach().cpu())
        self.val_targets.append(true_fahrenheit.detach().cpu())

        # DIRECT IMAGE LOGGING IN VALIDATION STEP
        # This works in both normal runs and sweeps
        if (batch_idx == 0 and  # Only first batch
                self.current_epoch % self.log_images_every_n_epochs == 0 and  # Every N epochs
                wandb.run is not None):  # Only if wandb is available

            try:
                print(f" Attempting to log images at epoch {self.current_epoch}")

                # Determine how many samples to log (up to 4, limited by batch size)
                batch_size = inputs.shape[0]
                num_samples_to_log = min(4, batch_size, self.max_images_to_log)

                # Convert to CPU and numpy for all samples we want to log
                inputs_cpu = inputs[0:num_samples_to_log].float().cpu().numpy()
                targets_cpu = targets[0:num_samples_to_log].float().cpu().numpy()
                predictions_cpu = predictions[0:num_samples_to_log].detach().float().cpu().numpy()

                # Create a list to store all the figures
                wandb_images = []

                # Process each sample
                for sample_idx in range(num_samples_to_log):
                    # Extract sequences for this sample (keeping original indexing logic)
                    input_seq = inputs_cpu[sample_idx]  # [time, H, W, channels]
                    target_seq = targets_cpu[sample_idx]  # [time, H, W, 1]
                    pred_seq = predictions_cpu[sample_idx]  # [time, H, W, 1]

                    input_len = input_seq.shape[0]
                    output_len = target_seq.shape[0]
                    max_timesteps = max(input_len, output_len)

                    # Create the visualization for this sample
                    fig, axes = plt.subplots(3, max_timesteps, figsize=(4 * max_timesteps, 12))

                    # Handle single timestep case
                    if max_timesteps == 1:
                        axes = axes.reshape(3, 1)

                    fig.patch.set_facecolor('lightgray')

                    # Row 0: Input sequences - check if LST exists in input channels
                    lst_band_idx, lst_exists = self._get_lst_band_index()

                    for t in range(input_len):
                        ax = axes[0, t]
                        ax.set_facecolor('lightgray')

                        if lst_exists == 'LST':
                            # LST exists - show it with color ramp and denormalize to Fahrenheit
                            lst_input = input_seq[t, :, :, lst_band_idx]  # Use LST channel
                            lst_input_fahrenheit = lst_input * (211.0 - (-189.0)) + (-189.0)

                            # Create mask for NODATA
                            nodata_mask = np.abs(lst_input_fahrenheit - (-189.0)) < 0.1
                            lst_masked = np.ma.masked_where(nodata_mask, lst_input_fahrenheit)

                            if not lst_masked.mask.all():
                                vmin_input = lst_masked.min()
                                vmax_input = lst_masked.max()
                                im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_input, vmax=vmax_input, alpha=0.9)
                                ax.set_title(f'Input LST T={t + 1}\n({vmin_input:.1f}°F - {vmax_input:.1f}°F)',
                                             fontsize=10)
                                plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                            else:
                                ax.imshow(np.zeros_like(lst_input_fahrenheit), cmap='RdYlBu_r', alpha=0)
                                ax.set_title(f'Input LST T={t + 1}\n(No Valid Data)', fontsize=10)
                        else:
                            # LST doesn't exist - show first channel (index 0) in grayscale, normalized
                            input_channel_0 = input_seq[t, :, :, 0]  # First channel, normalized

                            # Create mask for NODATA (value 0)
                            nodata_mask = input_channel_0 == 0
                            input_masked = np.ma.masked_where(nodata_mask, input_channel_0)

                            if not input_masked.mask.all():
                                vmin_input = input_masked.min()
                                vmax_input = input_masked.max()
                                im = ax.imshow(input_masked, cmap='gray', vmin=vmin_input, vmax=vmax_input, alpha=0.9)
                                ax.set_title(f'Input CH0 T={t + 1}\n({vmin_input:.3f} - {vmax_input:.3f})', fontsize=10)
                                plt.colorbar(im, ax=ax, fraction=0.046, label='Normalized')
                            else:
                                ax.imshow(np.zeros_like(input_channel_0), cmap='gray', alpha=0)
                                ax.set_title(f'Input CH0 T={t + 1}\n(No Valid Data)', fontsize=10)

                        ax.axis('off')

                    # Fill remaining input columns
                    for t in range(input_len, max_timesteps):
                        axes[0, t].set_facecolor('lightgray')
                        axes[0, t].axis('off')
                        axes[0, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0, t].transAxes)

                    # Row 1: Target sequences (always LST, denormalized to Fahrenheit)
                    for t in range(output_len):
                        ax = axes[1, t]
                        ax.set_facecolor('lightgray')

                        lst_target = target_seq[t, :, :, 0]
                        lst_target_fahrenheit = lst_target * (211.0 - (-189.0)) + (-189.0)

                        nodata_mask = np.abs(lst_target_fahrenheit - (-189.0)) < 0.1
                        lst_masked = np.ma.masked_where(nodata_mask, lst_target_fahrenheit)

                        if not lst_masked.mask.all():
                            vmin_target = lst_masked.min()
                            vmax_target = lst_masked.max()
                            im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_target, vmax=vmax_target, alpha=0.9)
                            ax.set_title(f'Target T={input_len + t + 1}\n({vmin_target:.1f}°F - {vmax_target:.1f}°F)',
                                         fontsize=10)
                            plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                        else:
                            ax.imshow(np.zeros_like(lst_target_fahrenheit), cmap='RdYlBu_r', alpha=0)
                            ax.set_title(f'Target T={input_len + t + 1}\n(No Valid Data)', fontsize=10)

                        ax.axis('off')

                    # Fill remaining target columns
                    for t in range(output_len, max_timesteps):
                        axes[1, t].set_facecolor('lightgray')
                        axes[1, t].axis('off')
                        axes[1, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, t].transAxes)

                    # Row 2: Prediction sequences (always LST, denormalized to Fahrenheit)
                    for t in range(output_len):
                        ax = axes[2, t]
                        ax.set_facecolor('lightgray')

                        lst_pred = pred_seq[t, :, :, 0]
                        lst_pred_fahrenheit = lst_pred * (211.0 - (-189.0)) + (-189.0)

                        # Use target's mask for predictions
                        target_lst = target_seq[t, :, :, 0] * (211.0 - (-189.0)) + (-189.0)
                        nodata_mask = np.abs(target_lst - (-189.0)) < 0.1
                        lst_masked = np.ma.masked_where(nodata_mask, lst_pred_fahrenheit)

                        if not lst_masked.mask.all():
                            vmin_pred = lst_masked.min()
                            vmax_pred = lst_masked.max()
                            im = ax.imshow(lst_masked, cmap='RdYlBu_r', vmin=vmin_pred, vmax=vmax_pred, alpha=0.9)
                            ax.set_title(f'Prediction T={input_len + t + 1}\n({vmin_pred:.1f}°F - {vmax_pred:.1f}°F)',
                                         fontsize=10)
                            plt.colorbar(im, ax=ax, fraction=0.046, label='°F')
                        else:
                            ax.imshow(np.zeros_like(lst_pred_fahrenheit), cmap='RdYlBu_r', alpha=0)
                            ax.set_title(f'Prediction T={input_len + t + 1}\n(No Valid Data)', fontsize=10)

                        ax.axis('off')

                    # Fill remaining prediction columns
                    for t in range(output_len, max_timesteps):
                        axes[2, t].set_facecolor('lightgray')
                        axes[2, t].axis('off')
                        axes[2, t].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2, t].transAxes)

                    # Add row labels - conditional based on LST availability
                    input_label = 'INPUT LST' if lst_exists == 'LST' else 'INPUT CH0'
                    axes[0, 0].text(-0.2, 0.5, input_label, rotation=90, ha='center', va='center',
                                    transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
                    axes[1, 0].text(-0.2, 0.5, 'TARGET LST', rotation=90, ha='center', va='center',
                                    transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
                    axes[2, 0].text(-0.2, 0.5, 'PREDICTED LST', rotation=90, ha='center', va='center',
                                    transform=axes[2, 0].transAxes, fontsize=12, fontweight='bold')

                    # Add title with sample information
                    plt.suptitle(
                        f'Validation Sample {sample_idx + 1}/{num_samples_to_log} - Epoch {self.current_epoch}, Batch {batch_idx}\n'
                        f'Input Length: {input_len}, Output Length: {output_len}', fontsize=12)
                    plt.tight_layout()

                    # Add this figure to our list
                    wandb_images.append(wandb.Image(fig))
                    plt.close(fig)

                    # Log all images at once
                    wandb.log({
                        "validation_predictions": wandb_images
                    }, step=self.global_step)

                    print(f"✅ Successfully logged validation images at epoch {self.current_epoch}")

            except Exception as e:
                print(f"❌ Image logging failed in validation_step: {e}")
                import traceback
                traceback.print_exc()

        return loss

    def _get_sample_metadata(self, batch_idx: int, sample_idx: int) -> dict:
        """Get metadata for a specific sample in the batch"""
        try:
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'train_dataset'):
                # Determine which dataset to use based on current stage
                if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'stage'):
                    if self.trainer.state.stage.name == 'VALIDATING':
                        dataset = self.trainer.datamodule.val_dataset
                    elif self.trainer.state.stage.name == 'TESTING':
                        dataset = getattr(self.trainer.datamodule, 'test_dataset', None)
                    else:
                        dataset = self.trainer.datamodule.train_dataset
                else:
                    # Fallback to train dataset
                    dataset = self.trainer.datamodule.train_dataset

                if dataset is None:
                    return {}

                # Calculate actual sample index from batch info
                batch_size = self.trainer.datamodule.batch_size
                actual_sample_idx = batch_idx * batch_size + sample_idx

                # Get the tile sequence info from dataset
                if hasattr(dataset, 'tile_sequences') and actual_sample_idx < len(dataset.tile_sequences):
                    city, tile_row, tile_col, input_months, output_months = dataset.tile_sequences[actual_sample_idx]

                    return {
                        'city': city,
                        'tile_position': f"row_{tile_row:03d}_col_{tile_col:03d}",
                        'tile_row': tile_row,
                        'tile_col': tile_col,
                        'input_months': input_months,
                        'output_months': output_months,
                        'input_date_range': f"{input_months[0]} to {input_months[-1]}",
                        'output_date_range': f"{output_months[0]} to {output_months[-1]}"
                    }

            return {}

        except Exception as e:
            print(f"Warning: Could not get sample metadata: {e}")
            return {}

    def _get_lst_band_index(self) -> int:
        """Get the index of the LST band in the current channel configuration"""
        removed_channels = []
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'remove_channels'):
            removed_channels = self.trainer.datamodule.remove_channels

        original_bands = ['DEM', 'LST', 'red', 'green', 'blue', 'ndvi', 'ndwi', 'ndbi', 'albedo']
        for channel_to_remove in removed_channels:
            original_bands.remove(channel_to_remove)
        if 'LST' in original_bands:
            return original_bands.index('LST'), 'LST'
        else:
            return 0, 'NO_LST'  # There was no LST!

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        self.eval()
        inputs, targets = batch

        # Get predictions from forward pass
        predictions = self(inputs)

        # Use heat_index_loss for earthnet_w_index model during test
        if self.use_heat_index_loss:
            loss = self.heat_index_loss(inputs, predictions, targets)
        else:
            loss = self.masked_loss(predictions, targets)

        with torch.no_grad():
            # Denormalize to Fahrenheit: value * (max - min) + min
            # Output is always temperature in Fahrenheit regardless of loss type
            pred_fahrenheit = predictions.detach() * (211.0 - (-189.0)) + (-189.0)
            true_fahrenheit = targets.detach() * (211.0 - (-189.0)) + (-189.0)
            self.test_predictions.append(pred_fahrenheit.detach().cpu())
            self.test_targets.append(true_fahrenheit.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        # Only calculate and log from rank 0 in distributed training
        if self.trainer.is_global_zero:
            if len(self.test_predictions) > 0 and len(self.test_targets) > 0:
                all_preds = torch.cat(self.test_predictions)
                all_targets = torch.cat(self.test_targets)
                test_mae_F = self.masked_mae(all_preds, all_targets)
                test_rmse_F = torch.sqrt(self.masked_loss(all_preds, all_targets))
                if wandb.run is not None:
                    wandb.log({
                        'test_mae_F': test_mae_F.item(),
                        'test_rmse_F': test_rmse_F.item()
                    })
        self.test_predictions = []
        self.test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )

        # More aggressive clipping for earthnet
        model_size = getattr(self.hparams, 'model_size', 'small')
        if model_size == "earthnet":
            grad_clip_val = 1.0
        elif model_size == "large":
            grad_clip_val = 1.0
        else:
            grad_clip_val = 2.0

        return {
            "optimizer": optimizer,
            "gradient_clip_val": grad_clip_val,
            "gradient_clip_algorithm": "norm"
        }

    def on_train_epoch_end(self):
        print("train epoch end ran")
        """Create correlation plot at end of training epoch"""
        # Only log from rank 0 in distributed training
        if self.trainer.is_global_zero and len(self.train_predictions) > 0 and wandb.run is not None:
            try:
                fig = self.create_correlation_plot(
                    self.train_predictions, self.train_targets,
                    "training", self.current_epoch
                )
                if fig is not None:
                    wandb.log({
                        "train_correlation_plot": wandb.Image(fig),
                        "epoch": self.current_epoch
                    })
                    plt.close(fig)
                    print(f"✅ Logged training correlation plot for epoch {self.current_epoch}")
            except Exception as e:
                print(f"❌ Failed to create training correlation plot: {e}")

        # Clear stored data on all ranks
        self.train_predictions = []
        self.train_targets = []

    def on_validation_epoch_end(self):
        print("validation epoch end ran")
        """Calculate epoch metrics and create correlation plot"""
        # Only log from rank 0 in distributed training
        if self.trainer.is_global_zero and len(self.val_predictions) > 0:
            # Concatenate all predictions and targets (already in Fahrenheit)
            all_preds = torch.cat(self.val_predictions)
            all_targets = torch.cat(self.val_targets)
            
            # Calculate metrics on ALL pixels (like test)
            val_mae_F_epoch = self.masked_mae(all_preds, all_targets)
            val_rmse_F_epoch = torch.sqrt(self.masked_loss(all_preds, all_targets))
            
            # Calculate correlation on all pixels (matching plot)
            pred_flat = all_preds.flatten()
            true_flat = all_targets.flatten()
            nodata_val = -189.0
            valid_mask = (true_flat > nodata_val + 1) & (pred_flat > nodata_val + 1)
            if valid_mask.sum() > 1:
                val_correlation_epoch = torch.corrcoef(torch.stack([pred_flat[valid_mask], true_flat[valid_mask]]))[0, 1]
            else:
                val_correlation_epoch = torch.tensor(0.0)
            
            # Log epoch metrics
            self.log('val_mae_F_epoch', val_mae_F_epoch, sync_dist=False)
            self.log('val_rmse_F_epoch', val_rmse_F_epoch, sync_dist=False)
            if torch.isfinite(val_correlation_epoch):
                self.log('val_correlation_epoch', val_correlation_epoch, sync_dist=False)
            
            if wandb.run is not None:
                try:
                    fig = self.create_correlation_plot(
                        self.val_predictions, self.val_targets,
                        "validation", self.current_epoch,
                        already_fahrenheit=True
                    )
                    if fig is not None:
                        wandb.log({
                            "val_correlation_plot": wandb.Image(fig),
                            "epoch": self.current_epoch
                        })
                        plt.close(fig)
                        print(f"✅ Logged validation correlation plot for epoch {self.current_epoch}")
                except Exception as e:
                    print(f"❌ Failed to create validation correlation plot: {e}")

        # Clear stored data on all ranks
        self.val_predictions = []
        self.val_targets = []
