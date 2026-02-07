"""
City-wide LST Inference for San Antonio
Downloads latest year of Landsat STAC data ONCE, runs inference on all tiles,
then generates static PNG map tiles and JSON temperature grids for GitHub Pages.
"""

import io
import json
import math
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, List, Tuple
import calendar
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from PIL import Image
from tqdm import tqdm

import pystac_client
import planetary_computer as pc
from odc.stac import stac_load

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAN_ANTONIO_BBOX = [-98.7459, 29.2534, -98.2467, 29.7534]
TILE_SIZE = 128
NODATA = 0
INPUT_BANDS = ["DEM", "LST", "red", "green", "blue", "ndvi", "ndwi", "ndbi", "albedo"]
BAND_RANGES = {
    "red": {"min": 1.0, "max": 10000.0},
    "ndwi": {"min": -10000.0, "max": 10000.0},
    "ndvi": {"min": -10000.0, "max": 10000.0},
    "ndbi": {"min": -10000.0, "max": 10000.0},
    "LST": {"min": -189.0, "max": 211.0},
    "green": {"min": 1.0, "max": 10000.0},
    "blue": {"min": 1.0, "max": 10000.0},
    "DEM": {"min": 9899.0, "max": 13110.0},
    "albedo": {"min": 1.0, "max": 9980.0},
}

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_PATH = SCRIPT_DIR / "model_baseline.ckpt"
SHAPEFILE_PATH = SCRIPT_DIR / "San_Antonio" / "San Antonio_TX.shp"
CLIP_GEOJSON = SCRIPT_DIR / "downtown_westside.geojson"

# ---------------------------------------------------------------------------
# Color ramp: teal -> yellow -> red, 25 discrete bins
# ---------------------------------------------------------------------------
NUM_BINS = 25
LUT = np.zeros((NUM_BINS, 3), dtype=np.uint8)
for _i in range(NUM_BINS):
    _t = _i / (NUM_BINS - 1)
    if _t < 0.5:
        _s = _t * 2
        LUT[_i] = (int(_s * 255), int(128 + _s * 127), int(128 - _s * 128))
    else:
        _s = (_t - 0.5) * 2
        LUT[_i] = (255, int(255 - _s * 255), 0)

# ---------------------------------------------------------------------------
# Tile math (EPSG:3857)
# ---------------------------------------------------------------------------
ORIGIN = 20037508.342789244
RENDER_TILE_SIZE = 256


def tile_bounds_3857(z, x, y):
    size = 2 * ORIGIN / (2 ** z)
    left = -ORIGIN + x * size
    right = left + size
    top = ORIGIN - y * size
    bottom = top - size
    return left, bottom, right, top


def robust_min_max(arr):
    """Find min/max of valid (non-NaN) pixels."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0, 0.0
    int_temps = valid.astype(int)
    unique_sorted = np.sort(np.unique(int_temps))
    return float(unique_sorted[0]), float(unique_sorted[-1])


def _bin_indices(data, valid_mask, vmin, vmax):
    """Map valid pixel values to bin indices 0..NUM_BINS-1, clamped."""
    idx = np.zeros(data.shape, dtype=np.uint8)
    if valid_mask.any():
        normed = (data[valid_mask] - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0, 1)
        idx[valid_mask] = np.clip((normed * (NUM_BINS - 1)).astype(int), 0, NUM_BINS - 1)
    return idx


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class CityWideInference:
    """City-wide LST inference + static tile generation."""

    def __init__(self):
        self.data_dir = SCRIPT_DIR
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stac_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        self.shapefile = gpd.read_file(SHAPEFILE_PATH)
        self.shapefile_3857 = self.shapefile.to_crs("EPSG:3857")
        self.scene_cache = {}
        self.dem_cache = None

        # Load clip polygon
        with open(CLIP_GEOJSON) as f:
            fc = json.load(f)
        proj = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        clip_4326 = shape(fc["features"][0]["geometry"])
        self.clip_poly = shapely_transform(proj.transform, clip_4326)
        self.clip_bounds = self.clip_poly.bounds

        # City-wide EPSG:3857 bounds
        self.cache_x_min, self.cache_y_min = proj.transform(SAN_ANTONIO_BBOX[0], SAN_ANTONIO_BBOX[1])
        self.cache_x_max, self.cache_y_max = proj.transform(SAN_ANTONIO_BBOX[2], SAN_ANTONIO_BBOX[3])

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        if self.model is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from model import LandsatLSTPredictor
            self.model = LandsatLSTPredictor.load_from_checkpoint(
                str(CHECKPOINT_PATH), map_location=self.device)
            self.model.eval()
            self.model.to(self.device)

    # ------------------------------------------------------------------
    # STAC scene selection
    # ------------------------------------------------------------------
    def get_best_landsat_scene(self, target_month: Optional[datetime] = None) -> Optional[Dict]:
        """Query STAC for the best Landsat 8/9 scene for San Antonio."""
        if target_month is None:
            target_month = datetime.now()
        year, month = target_month.year, target_month.month
        last_day = calendar.monthrange(year, month)[1]

        query = self.stac_client.search(
            collections=["landsat-c2-l2"], bbox=SAN_ANTONIO_BBOX,
            datetime=f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day}",
            query={"platform": {"in": ["landsat-8", "landsat-9"]}})

        items = list(query.items())
        if not items:
            return None

        scored_scenes = []
        for item in items:
            score_info = self._score_scene(item)
            if score_info is not None:
                scored_scenes.append((item, score_info))

        if not scored_scenes:
            scenes_with_cloud = [(item, item.properties.get("eo:cloud_cover", 100)) for item in items]
            scenes_with_cloud.sort(key=lambda x: x[1])
            best_item, cloud_cover = scenes_with_cloud[0]
            date_str = best_item.datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            return {"item": best_item, "date": date_str, "cloud_cover": cloud_cover}

        scored_scenes.sort(key=lambda x: x[1]["combined_score"], reverse=True)
        best_item, best_score = scored_scenes[0]
        date_str = best_item.datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        return {
            "item": best_item, "date": date_str,
            "cloud_cover": best_score["cloud_cover"],
            "coverage_pct": best_score["coverage_pct"],
            "quality_pct": best_score["quality_pct"],
            "combined_score": best_score["combined_score"],
        }

    def _score_scene(self, item) -> Optional[Dict]:
        """Score a scene based on cloud cover, coverage, and quality."""
        try:
            cloud_cover = item.properties.get("eo:cloud_cover", 100)
            preview = stac_load(
                [item], bands=["lwir11", "qa_pixel"], bbox=SAN_ANTONIO_BBOX,
                resolution=120, crs="EPSG:3857", skip_broken=True)

            if preview is None or "lwir11" not in preview:
                return {"cloud_cover": cloud_cover, "coverage_pct": 0,
                        "quality_pct": 0, "combined_score": 100 - cloud_cover}

            thermal = preview["lwir11"].values.squeeze()
            qa = preview["qa_pixel"].values.squeeze()
            total_pixels = thermal.size
            valid_pixels = np.sum(thermal > 0)
            coverage_pct = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            cloud_shadow_mask = (qa & (1 << 3)) != 0
            snow_mask = (qa & (1 << 4)) != 0
            cloud_mask = (qa & 0b00011000) != 0
            bad_pixels = np.sum(cloud_shadow_mask | snow_mask | cloud_mask)
            quality_pct = ((total_pixels - bad_pixels) / total_pixels) * 100 if total_pixels > 0 else 0

            combined_score = 0.40 * coverage_pct + 0.35 * quality_pct + 0.25 * (100 - cloud_cover)
            return {
                "cloud_cover": cloud_cover,
                "coverage_pct": round(coverage_pct, 1),
                "quality_pct": round(quality_pct, 1),
                "combined_score": round(combined_score, 1),
            }
        except Exception:
            cloud_cover = item.properties.get("eo:cloud_cover", 100)
            return {"cloud_cover": cloud_cover, "coverage_pct": 50,
                    "quality_pct": 50, "combined_score": 100 - cloud_cover}

    # ------------------------------------------------------------------
    # Band calculation helpers
    # ------------------------------------------------------------------
    def _calc_lst(self, thermal):
        kelvin = thermal * 0.00341802 + 149.0
        return (kelvin - 273.15) * 9 / 5 + 32

    def _to_refl(self, band):
        return np.clip(band * 0.0000275 - 0.2, 0, 1)

    def _calc_color(self, band):
        return self._to_refl(band) * 10000

    def _calc_ndvi(self, nir, red):
        n, r = self._to_refl(nir), self._to_refl(red)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num((n - r) / (n + r), 0), -1, 1) * 10000

    def _calc_ndwi(self, nir, green):
        n, g = self._to_refl(nir), self._to_refl(green)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num((g - n) / (g + n), 0), -1, 1) * 10000

    def _calc_ndbi(self, nir, swir):
        n, s = self._to_refl(nir), self._to_refl(swir)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.clip(np.nan_to_num((s - n) / (s + n), 0), -1, 1) * 10000

    def _calc_albedo(self, b, g, r, n, s):
        b, g, r, n, s = [np.clip(self._to_refl(x), 0, 1) for x in [b, g, r, n, s]]
        return np.clip(0.356 * b + 0.130 * g + 0.373 * r + 0.085 * n + 0.072 * s - 0.018, 0, 1) * 10000

    # ------------------------------------------------------------------
    # Tile grid / shapefile methods
    # ------------------------------------------------------------------
    def get_tile_grid_info(self) -> Tuple[int, int, rasterio.transform.Affine, Tuple]:
        tile_size_m = TILE_SIZE * 30
        n_cols = int(np.ceil((self.cache_x_max - self.cache_x_min) / tile_size_m))
        n_rows = int(np.ceil((self.cache_y_max - self.cache_y_min) / tile_size_m))
        transform = from_bounds(self.cache_x_min, self.cache_y_min,
                                self.cache_x_max, self.cache_y_max,
                                n_cols * TILE_SIZE, n_rows * TILE_SIZE)
        return n_rows, n_cols, transform, (self.cache_x_min, self.cache_y_min,
                                           self.cache_x_max, self.cache_y_max)

    def tile_intersects_shapefile(self, row: int, col: int, bounds: Tuple) -> bool:
        from shapely.geometry import box
        x_min, y_min, x_max, y_max = bounds
        tile_size_m = TILE_SIZE * 30
        tx_min = x_min + col * tile_size_m
        ty_max = y_max - row * tile_size_m
        tile_box = box(tx_min, ty_max - tile_size_m, tx_min + tile_size_m, ty_max)
        return self.shapefile_3857.geometry.iloc[0].intersects(tile_box)

    def get_valid_tiles(self) -> List[Tuple[int, int]]:
        n_rows, n_cols, _, bounds = self.get_tile_grid_info()
        valid = []
        for r in range(n_rows):
            for c in range(n_cols):
                if self.tile_intersects_shapefile(r, c, bounds):
                    valid.append((r, c))
        return valid

    # ------------------------------------------------------------------
    # Scene downloading
    # ------------------------------------------------------------------
    def download_all_scenes(self, num_months: int = 12):
        print(f"\nDownloading {num_months} months of scenes for entire city...")
        current = datetime.now()
        months_to_fetch = []
        for i in range(num_months):
            target = current - relativedelta(months=i + 1)
            months_to_fetch.append(target)
        months_to_fetch.reverse()

        print("Fetching DEM...")
        self._download_dem()

        for target_month in tqdm(months_to_fetch, desc="Downloading scenes"):
            month_key = target_month.strftime("%Y-%m")
            cache_dir = self.data_dir / "scene_cache" / month_key
            if cache_dir.exists() and (cache_dir / "LST.npy").exists():
                print(f"  {month_key}: Loading from cache")
                self._load_scene_from_cache(month_key, cache_dir)
                continue

            scene_info = self.get_best_landsat_scene(target_month)
            if scene_info is None:
                print(f"  {month_key}: No scene found, will interpolate")
                continue

            score = scene_info.get('combined_score', 'N/A')
            cloud = scene_info['cloud_cover']
            print(f"  {month_key}: {scene_info['date'][:10]} (score={score}, cloud={cloud:.1f}%)")
            self._download_and_process_full_scene(month_key, scene_info)

        self._interpolate_missing_months(months_to_fetch)

    def _download_dem(self):
        cache_path = self.data_dir / "scene_cache" / "DEM.npy"
        if cache_path.exists():
            self.dem_cache = np.load(cache_path)
            return
        try:
            dem_query = self.stac_client.search(collections=["nasadem"], bbox=SAN_ANTONIO_BBOX)
            dem_rasters = stac_load(
                list(dem_query.items()), bands=["elevation"], bbox=SAN_ANTONIO_BBOX,
                resolution=30, crs="EPSG:3857", skip_broken=True, fail_on_error=True)
            dem = dem_rasters["elevation"].values.squeeze() + 10000
            self.dem_cache = self._normalize_band(dem, "DEM")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.dem_cache)
        except Exception as e:
            print(f"Error fetching DEM: {e}")

    def _download_and_process_full_scene(self, month_key: str, scene_info: Dict):
        item = scene_info["item"]
        cache_dir = self.data_dir / "scene_cache" / month_key
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            landsat_rasters = stac_load(
                [item], bands=["blue", "green", "red", "nir08", "swir16", "lwir11", "qa_pixel"],
                bbox=SAN_ANTONIO_BBOX, resolution=30, crs="EPSG:3857",
                skip_broken=True, fail_on_error=True)
            scene = landsat_rasters.isel(time=0)
            thermal = scene["lwir11"].values
            red, green, blue = scene["red"].values, scene["green"].values, scene["blue"].values
            nir, swir, qa = scene["nir08"].values, scene["swir16"].values, scene["qa_pixel"].values
            valid_mask = (thermal > 0) & ((qa & 0b00011000) == 0)

            bands = {}
            bands["LST"] = self._normalize_band(np.where(valid_mask, self._calc_lst(thermal), NODATA), "LST")
            bands["red"] = self._normalize_band(np.where(valid_mask, self._calc_color(red), NODATA), "red")
            bands["green"] = self._normalize_band(np.where(valid_mask, self._calc_color(green), NODATA), "green")
            bands["blue"] = self._normalize_band(np.where(valid_mask, self._calc_color(blue), NODATA), "blue")
            bands["ndvi"] = self._normalize_band(np.where(valid_mask, self._calc_ndvi(nir, red), NODATA), "ndvi")
            bands["ndwi"] = self._normalize_band(np.where(valid_mask, self._calc_ndwi(nir, green), NODATA), "ndwi")
            bands["ndbi"] = self._normalize_band(np.where(valid_mask, self._calc_ndbi(nir, swir), NODATA), "ndbi")
            bands["albedo"] = self._normalize_band(
                np.where(valid_mask, self._calc_albedo(blue, green, red, nir, swir), NODATA), "albedo")

            self.scene_cache[month_key] = bands
            for band_name, band_data in bands.items():
                np.save(cache_dir / f"{band_name}.npy", band_data)
        except Exception as e:
            print(f"  Error downloading {month_key}: {e}")

    def _load_scene_from_cache(self, month_key: str, cache_dir: Path):
        bands = {}
        for band_name in INPUT_BANDS[1:]:
            band_path = cache_dir / f"{band_name}.npy"
            if band_path.exists():
                bands[band_name] = np.load(band_path)
        self.scene_cache[month_key] = bands

    def _interpolate_missing_months(self, months: List[datetime]):
        month_keys = [m.strftime("%Y-%m") for m in months]
        available = [k for k in month_keys if k in self.scene_cache]
        if not available:
            print("WARNING: No scenes available!")
            return
        for key in month_keys:
            if key not in self.scene_cache:
                nearest = min(available, key=lambda x: abs(
                    datetime.strptime(x, "%Y-%m") - datetime.strptime(key, "%Y-%m")))
                self.scene_cache[key] = self.scene_cache[nearest]
                print(f"  {key}: Interpolated from {nearest}")

    def _normalize_band(self, data: np.ndarray, band_name: str) -> np.ndarray:
        rng = BAND_RANGES[band_name]
        mask = data != NODATA
        norm = np.zeros_like(data, dtype=np.float32)
        norm[mask] = (data[mask] - rng["min"]) / (rng["max"] - rng["min"])
        return np.clip(norm, 0, 1)

    # ------------------------------------------------------------------
    # Tile extraction + inference
    # ------------------------------------------------------------------
    def extract_tile_sequence(self, tile_row: int, tile_col: int, num_months: int = 12) -> Optional[np.ndarray]:
        current = datetime.now()
        months = []
        for i in range(num_months):
            target = current - relativedelta(months=i + 1)
            months.append(target.strftime("%Y-%m"))
        months.reverse()

        sequence = []
        for month_key in months:
            if month_key not in self.scene_cache:
                return None
            tile_bands = [self._extract_tile(self.dem_cache, tile_row, tile_col)]
            scene_bands = self.scene_cache[month_key]
            for band_name in INPUT_BANDS[1:]:
                if band_name not in scene_bands:
                    return None
                tile = self._extract_tile(scene_bands[band_name], tile_row, tile_col)
                if tile is None:
                    return None
                tile_bands.append(tile)
            sequence.append(np.stack(tile_bands, axis=-1))
        return np.stack(sequence, axis=0)

    def _extract_tile(self, data: np.ndarray, row: int, col: int) -> Optional[np.ndarray]:
        sr, sc = row * TILE_SIZE, col * TILE_SIZE
        if sr + TILE_SIZE > data.shape[0] or sc + TILE_SIZE > data.shape[1]:
            tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=data.dtype)
            h = min(TILE_SIZE, data.shape[0] - sr)
            w = min(TILE_SIZE, data.shape[1] - sc)
            if h > 0 and w > 0:
                tile[:h, :w] = data[sr:sr + h, sc:sc + w]
            return tile
        return data[sr:sr + TILE_SIZE, sc:sc + TILE_SIZE].copy()

    def _apply_shapefile_mask(self, data: np.ndarray, transform, bounds) -> np.ndarray:
        x_min, y_min, x_max, y_max = bounds
        full_transform = from_bounds(x_min, y_min, x_max, y_max, data.shape[1], data.shape[0])
        mask = geometry_mask(self.shapefile_3857.geometry, out_shape=data.shape,
                             transform=full_transform, invert=True)
        return np.where(mask, data, np.nan)

    def run_city_inference(self) -> np.ndarray:
        """Run inference on all valid tiles, return prediction in Fahrenheit."""
        print("Loading model...")
        self.load_model()

        n_rows, n_cols, transform, bounds = self.get_tile_grid_info()
        valid_tiles = self.get_valid_tiles()
        print(f"Found {len(valid_tiles)} tiles within city boundary (out of {n_rows}x{n_cols} grid)")

        self.download_all_scenes(num_months=12)

        output_height = n_rows * TILE_SIZE
        output_width = n_cols * TILE_SIZE
        prediction_map = np.full((output_height, output_width), np.nan, dtype=np.float32)

        print("\nRunning inference on tiles...")
        successful = failed = 0
        for row, col in tqdm(valid_tiles, desc="Processing tiles"):
            try:
                input_seq = self.extract_tile_sequence(row, col, num_months=12)
                if input_seq is None:
                    failed += 1
                    continue
                input_tensor = torch.from_numpy(input_seq).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pred = self.model(input_tensor)
                pred_f = pred.cpu().numpy().squeeze() * 400.0 - 189.0
                sr, sc = row * TILE_SIZE, col * TILE_SIZE
                prediction_map[sr:sr + TILE_SIZE, sc:sc + TILE_SIZE] = pred_f
                successful += 1
            except Exception as e:
                print(f"\n  Error on tile ({row}, {col}): {e}")
                failed += 1

        print(f"\nProcessed {successful} tiles successfully, {failed} failed")
        print("Applying city boundary mask...")
        return self._apply_shapefile_mask(prediction_map, transform, bounds)

    # ------------------------------------------------------------------
    # Static tile + JSON generation
    # ------------------------------------------------------------------
    def _clip_mask(self, left, bottom, right, top):
        """Boolean mask (True = inside clip polygon) for a 256x256 tile grid."""
        tfm = from_bounds(left, bottom, right, top, RENDER_TILE_SIZE, RENDER_TILE_SIZE)
        outside = geometry_mask([self.clip_poly], out_shape=(RENDER_TILE_SIZE, RENDER_TILE_SIZE),
                                transform=tfm, invert=False)
        return ~outside

    def _render_array_tile(self, data_arr, left, bottom, right, top, vmin, vmax):
        """Render a colored RGBA PNG tile from a source array. Returns bytes or None."""
        h, w = data_arr.shape
        tfm = from_bounds(self.cache_x_min, self.cache_y_min,
                          self.cache_x_max, self.cache_y_max, w, h)

        cl = max(left, self.cache_x_min)
        cb = max(bottom, self.cache_y_min)
        cr = min(right, self.cache_x_max)
        ct = min(top, self.cache_y_max)

        inv = ~tfm
        px_left, py_top = inv * (cl, ct)
        px_right, py_bottom = inv * (cr, cb)
        px_left, px_right = sorted([px_left, px_right])
        py_top, py_bottom = sorted([py_top, py_bottom])

        c0 = int(max(0, math.floor(px_left)))
        r0 = int(max(0, math.floor(py_top)))
        c1 = int(min(w, math.ceil(px_right)))
        r1 = int(min(h, math.ceil(py_bottom)))
        if c1 <= c0 or r1 <= r0:
            return None

        chip = data_arr[r0:r1, c0:c1].copy()
        nan_mask = np.isnan(chip)
        chip[nan_mask] = -9999.0

        img_chip = Image.fromarray(chip, mode="F")
        img_resized = img_chip.resize((RENDER_TILE_SIZE, RENDER_TILE_SIZE), Image.BILINEAR)
        data = np.array(img_resized)

        inside = self._clip_mask(left, bottom, right, top)
        valid_mask = (data > -9000.0) & inside
        if not valid_mask.any():
            return None

        idx = _bin_indices(data, valid_mask, vmin, vmax)
        rgba = np.zeros((RENDER_TILE_SIZE, RENDER_TILE_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = LUT[idx]
        rgba[valid_mask, 3] = 200

        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _render_raster_tile(self, raster_path, left, bottom, right, top, vmin, vmax):
        """Render a tile directly from a GeoTIFF prediction raster."""
        from rasterio.windows import from_bounds as window_from_bounds
        from rasterio.enums import Resampling

        with rasterio.open(raster_path) as ds:
            bounds = ds.bounds
            window = window_from_bounds(
                max(left, bounds.left), max(bottom, bounds.bottom),
                min(right, bounds.right), min(top, bounds.top),
                ds.transform)
            data = ds.read(1, window=window, out_shape=(RENDER_TILE_SIZE, RENDER_TILE_SIZE),
                           resampling=Resampling.bilinear)

        inside = self._clip_mask(left, bottom, right, top)
        valid_mask = ~np.isnan(data) & inside
        if not valid_mask.any():
            return None

        idx = _bin_indices(data, valid_mask, vmin, vmax)
        rgba = np.zeros((RENDER_TILE_SIZE, RENDER_TILE_SIZE, 4), dtype=np.uint8)
        rgba[..., :3] = LUT[idx]
        rgba[valid_mask, 3] = 200

        img = Image.fromarray(rgba, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _masked_robust(self, arr, tfm):
        """Compute robust_min_max only for pixels inside the clip polygon."""
        outside = geometry_mask([self.clip_poly], out_shape=arr.shape,
                                transform=tfm, invert=False)
        clipped = arr.copy()
        clipped[outside] = np.nan
        return robust_min_max(clipped)

    def generate_static_site(self, prediction_map: np.ndarray):
        """Generate tiles/, temperature/, and months.json from prediction + scene cache."""
        tiles_dir = SCRIPT_DIR / "tiles"
        temp_dir = SCRIPT_DIR / "temperature"
        tiles_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)

        # Build monthly Fahrenheit arrays from scene cache
        monthly_f = {}
        for month_key, bands in self.scene_cache.items():
            if "LST" not in bands:
                continue
            arr = bands["LST"]
            fahrenheit = arr.astype(np.float32) * 400.0 - 189.0
            fahrenheit[arr == 0] = np.nan
            monthly_f[month_key] = fahrenheit

        # Compute temperature ranges for each dataset
        available_months = sorted(monthly_f.keys())
        ranges = {}

        cache_tfm = from_bounds(self.cache_x_min, self.cache_y_min,
                                self.cache_x_max, self.cache_y_max,
                                monthly_f[available_months[0]].shape[1],
                                monthly_f[available_months[0]].shape[0])
        for m in available_months:
            rmin, rmax = self._masked_robust(monthly_f[m], cache_tfm)
            ranges[m] = {"temp_min_f": round(rmin), "temp_max_f": round(rmax)}
            print(f"  {m}: {rmin:.0f}F - {rmax:.0f}F")

        _, _, _, bounds = self.get_tile_grid_info()
        pred_tfm = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3],
                               prediction_map.shape[1], prediction_map.shape[0])
        pred_min, pred_max = self._masked_robust(prediction_map, pred_tfm)
        ranges["prediction"] = {"temp_min_f": round(pred_min), "temp_max_f": round(pred_max)}
        print(f"  prediction: {pred_min:.0f}F - {pred_max:.0f}F")

        # Save prediction as temp GeoTIFF for tile rendering
        pred_tif = SCRIPT_DIR / "city_inference_output" / "_temp_prediction.tif"
        pred_tif.parent.mkdir(exist_ok=True)
        with rasterio.open(
            str(pred_tif), "w", driver="GTiff",
            height=prediction_map.shape[0], width=prediction_map.shape[1],
            count=1, dtype=prediction_map.dtype, crs="EPSG:3857",
            transform=pred_tfm, compress="LZW", nodata=np.nan
        ) as dst:
            dst.write(prediction_map, 1)

        # Write months.json
        months_meta = {
            "months": available_months,
            "ranges": ranges,
            "num_bins": NUM_BINS,
        }
        with open(SCRIPT_DIR / "months.json", "w") as f:
            json.dump(months_meta, f)
        print(f"Wrote months.json ({len(available_months)} months + prediction)")

        # Generate PNG tiles at z14-17
        all_month_keys = available_months + ["prediction"]
        zoom_levels = [14, 15, 16, 17]

        trans_img = Image.new("RGBA", (RENDER_TILE_SIZE, RENDER_TILE_SIZE), (0, 0, 0, 0))
        trans_buf = io.BytesIO()
        trans_img.save(trans_buf, format="PNG")
        transparent_bytes = trans_buf.getvalue()

        total_tiles = 0
        for z in zoom_levels:
            size = 2 * ORIGIN / (2 ** z)
            x_start = int(math.floor((self.clip_bounds[0] + ORIGIN) / size))
            x_end = int(math.floor((self.clip_bounds[2] + ORIGIN) / size))
            y_start = int(math.floor((ORIGIN - self.clip_bounds[3]) / size))
            y_end = int(math.floor((ORIGIN - self.clip_bounds[1]) / size))

            tile_coords = [(x, y) for x in range(x_start, x_end + 1)
                           for y in range(y_start, y_end + 1)]

            for month_key in all_month_keys:
                vmin = ranges[month_key]["temp_min_f"]
                vmax = ranges[month_key]["temp_max_f"]

                if month_key == "prediction":
                    data_arr = None
                    raster_path = str(pred_tif)
                else:
                    data_arr = monthly_f.get(month_key)
                    raster_path = None

                for x, y in tile_coords:
                    left, bottom, right, top = tile_bounds_3857(z, x, y)

                    if (right < self.clip_bounds[0] or left > self.clip_bounds[2] or
                            top < self.clip_bounds[1] or bottom > self.clip_bounds[3]):
                        png_bytes = transparent_bytes
                    elif month_key == "prediction":
                        png_bytes = self._render_raster_tile(
                            raster_path, left, bottom, right, top, vmin, vmax)
                        if png_bytes is None:
                            png_bytes = transparent_bytes
                    else:
                        png_bytes = self._render_array_tile(
                            data_arr, left, bottom, right, top, vmin, vmax)
                        if png_bytes is None:
                            png_bytes = transparent_bytes

                    out_path = tiles_dir / month_key / str(z) / str(x) / f"{y}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "wb") as f:
                        f.write(png_bytes)
                    total_tiles += 1

            print(f"  z{z}: {len(tile_coords)} tiles/month x {len(all_month_keys)} months")

        print(f"Generated {total_tiles} total tiles")

        # Generate temperature JSON grids
        for month_key in all_month_keys:
            if month_key == "prediction":
                arr = prediction_map
                x_min, y_min, x_max, y_max = bounds
            else:
                arr = monthly_f[month_key]
                x_min = self.cache_x_min
                y_min = self.cache_y_min
                x_max = self.cache_x_max
                y_max = self.cache_y_max

            cb = self.clip_bounds
            h, w = arr.shape
            tfm = from_bounds(x_min, y_min, x_max, y_max, w, h)
            inv = ~tfm

            px_left, py_top = inv * (cb[0], cb[3])
            px_right, py_bottom = inv * (cb[2], cb[1])
            px_left, px_right = sorted([px_left, px_right])
            py_top, py_bottom = sorted([py_top, py_bottom])

            c0 = int(max(0, math.floor(px_left)))
            r0 = int(max(0, math.floor(py_top)))
            c1 = int(min(w, math.ceil(px_right)))
            r1 = int(min(h, math.ceil(py_bottom)))

            sub = arr[r0:r1, c0:c1].copy()

            sub_tfm = from_bounds(cb[0], cb[1], cb[2], cb[3], sub.shape[1], sub.shape[0])
            outside = geometry_mask([self.clip_poly], out_shape=sub.shape,
                                    transform=sub_tfm, invert=False)
            sub[outside] = np.nan

            data_list = []
            for row in sub:
                data_list.append([round(float(v), 1) if not np.isnan(v) else None for v in row])

            grid = {
                "bbox_3857": [cb[0], cb[1], cb[2], cb[3]],
                "cols": sub.shape[1],
                "rows": sub.shape[0],
                "cell_size": 30,
                "data": data_list,
            }

            out_file = temp_dir / f"{month_key}.json"
            with open(out_file, "w") as f:
                json.dump(grid, f)
            size_kb = out_file.stat().st_size / 1024
            print(f"  {month_key}.json: {sub.shape[0]}x{sub.shape[1]}, {size_kb:.0f}KB")

        pred_tif.unlink(missing_ok=True)
        print("Static site generation complete!")


def main():
    print("=" * 60)
    print("San Antonio City-Wide LST Prediction + Static Site Generation")
    print("=" * 60)

    inference = CityWideInference()
    prediction = inference.run_city_inference()

    valid_data = prediction[~np.isnan(prediction)]
    if len(valid_data) > 0:
        print(f"\nPrediction Statistics:")
        print(f"  Min: {np.min(valid_data):.1f} F")
        print(f"  Max: {np.max(valid_data):.1f} F")
        print(f"  Mean: {np.mean(valid_data):.1f} F")
        print(f"  Valid pixels: {len(valid_data):,}")
    else:
        print("\nNo valid predictions generated.")
        return

    print("\nGenerating static site assets...")
    inference.generate_static_site(prediction)


if __name__ == "__main__":
    main()
