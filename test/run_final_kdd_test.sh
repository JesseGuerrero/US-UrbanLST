#!/bin/bash

echo "Running ablation + earthnet test evaluation..."

# =============================================
# CHANNEL ABLATION (LSTM models)
# =============================================

# valiant-river-73: removed LST (8 channels)
echo "=== Test 1/22: valiant-river-73 (removed LST) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels LST --checkpoint ../test-bed/valiant-river-73/epoch=04.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 2/22: valiant-river-73 (removed LST) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels LST --checkpoint ../test-bed/valiant-river-73/epoch=04.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

# dark-sunset-72: LST only (1 channel)
echo "=== Test 3/22: dark-sunset-72 (LST only) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels DEM red green blue ndvi ndwi ndbi albedo --checkpoint ../test-bed/dark-sunset-72/epoch=03.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 4/22: dark-sunset-72 (LST only) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels DEM red green blue ndvi ndwi ndbi albedo --checkpoint ../test-bed/dark-sunset-72/epoch=03.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

# mild-music-71: RGB only (3 channels)
echo "=== Test 5/22: mild-music-71 (RGB only) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels DEM LST ndvi ndwi ndbi albedo --checkpoint ../test-bed/mild-music-71/epoch=05.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 6/22: mild-music-71 (RGB only) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels DEM LST ndvi ndwi ndbi albedo --checkpoint ../test-bed/mild-music-71/epoch=05.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

# =============================================
# EARTHNET CLUSTER MODELS (lr=0.0001, 9 channels)
# =============================================

# call-lucky-water-86: cluster=all
echo "=== Test 7/22: call-lucky-water-86 (earthnet, cluster=all) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/call-lucky-water-86/epoch=02.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 8/22: call-lucky-water-86 (earthnet, cluster=all) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/call-lucky-water-86/epoch=02.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# c1-cosmic-waterfall-85: cluster=1
echo "=== Test 9/22: c1-cosmic-waterfall-85 (earthnet, cluster=1) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 1 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/c1-cosmic-waterfall-85/epoch=11.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 10/22: c1-cosmic-waterfall-85 (earthnet, cluster=1) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 1 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/c1-cosmic-waterfall-85/epoch=11.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# c2-spring-darkness-87: cluster=2
echo "=== Test 11/22: c2-spring-darkness-87 (earthnet, cluster=2) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 2 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/c2-spring-darkness-87/epoch=14.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 12/22: c2-spring-darkness-87 (earthnet, cluster=2) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 2 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/c2-spring-darkness-87/epoch=14.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# c3-fallen-fire-88: cluster=3
echo "=== Test 13/22: c3-fallen-fire-88 (earthnet, cluster=3) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 3 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/c3-fallen-fire-88/epoch=24.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 14/22: c3-fallen-fire-88 (earthnet, cluster=3) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 3 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/c3-fallen-fire-88/epoch=24.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# C4-woven-music-89: cluster=4
echo "=== Test 15/22: C4-woven-music-89 (earthnet, cluster=4) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 4 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/C4-woven-music-89/epoch=05.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 16/22: C4-woven-music-89 (earthnet, cluster=4) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 4 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/C4-woven-music-89/epoch=05.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# =============================================
# EARTHNET CHANNEL ABLATION (from paper Table 5)
# =============================================

# cerulean-deluge-44: earthnet, removed LST (Spectral, 8 channels) - RMSE 22.29F
echo "=== Test 17/22: cerulean-deluge-44 (earthnet, Spectral/no LST) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels LST --checkpoint ../test-bed/cerulean-deluge-44/epoch=03.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 18/22: cerulean-deluge-44 (earthnet, Spectral/no LST) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels LST --checkpoint ../test-bed/cerulean-deluge-44/epoch=03.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# laced-deluge-43: earthnet, LST only (1 channel) - RMSE 23.45F
echo "=== Test 19/22: laced-deluge-43 (earthnet, LST only) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels DEM red green blue ndvi ndwi ndbi albedo --checkpoint ../test-bed/laced-deluge-43/epoch=10.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 20/22: laced-deluge-43 (earthnet, LST only) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels DEM red green blue ndvi ndwi ndbi albedo --checkpoint ../test-bed/laced-deluge-43/epoch=10.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

# swift-waterfall-4: earthnet, RGB only (3 channels) - RMSE 15.62F
echo "=== Test 21/22: swift-waterfall-4 (earthnet, RGB only) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --remove_channels DEM LST ndvi ndwi ndbi albedo --checkpoint ../test-bed/swift-waterfall-4/epoch=02.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 22/22: swift-waterfall-4 (earthnet, RGB only) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --remove_channels DEM LST ndvi ndwi ndbi albedo --checkpoint ../test-bed/swift-waterfall-4/epoch=02.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size earthnet

echo "=== Test 9/18: lstm-genial-sky-7 (CNN-LSTM, cluster=all) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 10/18: lstm-genial-sky-7 (CNN-LSTM, cluster=all) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster all --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 11/18: lstm-genial-sky-7 (CNN-LSTM, cluster=1) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 1 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 12/18: lstm-genial-sky-7 (CNN-LSTM, cluster=1) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 1 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 13/18: lstm-genial-sky-7 (CNN-LSTM, cluster=2) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 2 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 14/18: lstm-genial-sky-7 (CNN-LSTM, cluster=2) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 2 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 15/18: lstm-genial-sky-7 (CNN-LSTM, cluster=3) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 3 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 16/18: lstm-genial-sky-7 (CNN-LSTM, cluster=3) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 3 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 17/18: lstm-genial-sky-7 (CNN-LSTM, cluster=4) - Val set ==="
python run_test.py --dataset_root ../Data/ML --cluster 4 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2022 2023 --test_years 2022 2023 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm

echo "=== Test 18/18: lstm-genial-sky-7 (CNN-LSTM, cluster=4) - Test set ==="
python run_test.py --dataset_root ../Data/ML --cluster 4 --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --val_years 2024 2025 --test_years 2024 2025 --max_nodata 0.5 --checkpoint ../test-bed/lstm-genial-sky-7/all.ckpt --wandb_project AAAI-Project-channel-ablation-test --num_workers 32 --gpus 2 --precision 32 --model_size lstm


echo "All tests completed!"
