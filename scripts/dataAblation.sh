#!/bin/bash

echo " Building cache for all ablation experiment..."
echo ""

# Base parameters
DATASET_ROOT="./Data/ML"
OUTPUT_LENGTH=1
TRAIN_YEARS="2013 2014 2015 2016 2017 2018 2019 2020 2021"
VAL_YEARS="2022 2023"
TEST_YEARS="2024 2025"

# Single city configuration
INPUT_LENGTH=12
MAX_NODATA=0.5

# Channel ablation configurations (must match dataAblation.sh exactly)
REMOVED_CHANNELS_CONFIGS=(
    "DEM LST ndvi ndwi ndbi albedo"
    "DEM red green blue ndvi ndwi ndbi albedo"
    "LST"
)

echo "================================================="
echo " Building cache for single all ablation experiment"
echo "   City: $CITY"
echo "   Input Length: $INPUT_LENGTH"
echo "   Max NoData: $MAX_NODATA"
echo "================================================="

for removed_channels in "${REMOVED_CHANNELS_CONFIGS[@]}"; do
  CURRENT=$((CURRENT + 1))
  python setup_data.py \
      --remove_channels $removed_channels \
      --dataset_root "$DATASET_ROOT" \
      --cluster "all" \
      --input_length $INPUT_LENGTH \
      --output_length $OUTPUT_LENGTH \
      --train_years $TRAIN_YEARS \
      --val_years $VAL_YEARS \
      --test_years $TEST_YEARS \
      --max_nodata $MAX_NODATA \
      --max_output_nodata 0.0
done
if [ $? -eq 0 ]; then
    echo " Cache building complete for San Antonio!"
else
    echo "❌ Cache building failed"
    exit 1
fi

echo ""
echo " You can now run: bash trainSanAntonio.sh"
