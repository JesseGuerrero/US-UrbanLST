#!/bin/bash

echo "Building caches for all ablation test configurations..."
echo "Only building test split caches (trainer.test only needs stage='test')"
echo ""

# Common params
COMMON="--dataset_root ./Data/ML --input_length 12 --output_length 1 --train_years 2013 2014 2015 2016 2017 2018 2019 2020 2021 --max_nodata 0.5 --splits test"

# =============================================
# CONFIG A: val/test = 2022-2023 (Val set runs)
# =============================================
YEARS_A="--val_years 2022 2023 --test_years 2022 2023"

echo "=== Cache 2/16: cluster=1, no remove, 2022-2023 ==="
python setup_data.py $COMMON $YEARS_A --cluster 1

echo "=== Cache 3/16: cluster=2, no remove, 2022-2023 ==="
python setup_data.py $COMMON $YEARS_A --cluster 2

echo "=== Cache 4/16: cluster=3, no remove, 2022-2023 ==="
python setup_data.py $COMMON $YEARS_A --cluster 3

echo "=== Cache 5/16: cluster=4, no remove, 2022-2023 ==="
python setup_data.py $COMMON $YEARS_A --cluster 4

# =============================================
# CONFIG B: val/test = 2024-2025 (Test set runs)
# =============================================
YEARS_B="--val_years 2024 2025 --test_years 2024 2025"

echo "=== Cache 10/16: cluster=1, no remove, 2024-2025 ==="
python setup_data.py $COMMON $YEARS_B --cluster 1

echo "=== Cache 11/16: cluster=2, no remove, 2024-2025 ==="
python setup_data.py $COMMON $YEARS_B --cluster 2

echo "=== Cache 12/16: cluster=3, no remove, 2024-2025 ==="
python setup_data.py $COMMON $YEARS_B --cluster 3

echo "=== Cache 13/16: cluster=4, no remove, 2024-2025 ==="
python setup_data.py $COMMON $YEARS_B --cluster 4

echo ""
echo "All caches built! You can now run: cd test && bash run_ablation_test.sh"
