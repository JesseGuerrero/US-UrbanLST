#!/bin/bash

# Training Script for San Antonio, TX single-city experiment
# Uses Earthformer model with standard train/val/test splits

LOG_FILE="train_san_antonio.txt"

log_and_echo() {
    echo "$1" | tee -a "$LOG_FILE"
}

log_and_echo " Running training for San Antonio, TX single-city experiment..."
log_and_echo "Using Earthformer model"
log_and_echo ""

# Base parameters
DATASET_ROOT="./Data/ML"
OUTPUT_LENGTH=1
TRAIN_YEARS="2013 2014 2015 2016 2017 2018 2019 2020 2021"
VAL_YEARS="2022 2023"
TEST_YEARS="2024 2025"

# Single city configuration
CITY="San Antonio_TX"
INPUT_LENGTH=12
MAX_NODATA=0.5

# Training parameters
WANDB_PROJECT="AAAI-Project-SanAntonio-single-city"
LEARNING_RATE=0.00005
BATCH_SIZE=32
MAX_EPOCHS=200
NUM_WORKERS=32
GPUS=2
PRECISION=32
MODEL_SIZE="earthnet_w_index"
LIMIT_TRAIN_BATCHES=1.0
LIMIT_VAL_BATCHES=1.0
LIMIT_TEST_BATCHES=1.0

log_and_echo "================================================="
log_and_echo " Starting training"
log_and_echo "   City: $CITY"
log_and_echo "   Input Length: $INPUT_LENGTH"
log_and_echo "   Max NoData: $MAX_NODATA"
log_and_echo "   Learning Rate: $LEARNING_RATE"
log_and_echo "   Precision: $PRECISION"
log_and_echo "   Model: $MODEL_SIZE (Earthformer)"
log_and_echo "================================================="

python train_with_cache.py \
    --dataset_root "$DATASET_ROOT" \
    --cluster "all" \
    --input_length $INPUT_LENGTH \
    --output_length $OUTPUT_LENGTH \
    --train_years $TRAIN_YEARS \
    --val_years $VAL_YEARS \
    --test_years $TEST_YEARS \
    --max_nodata $MAX_NODATA \
    --max_output_nodata 0.25 \
    --wandb_project "$WANDB_PROJECT" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --num_workers $NUM_WORKERS \
    --gpus $GPUS \
    --precision $PRECISION \
    --model_size "$MODEL_SIZE" \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --limit_val_batches $LIMIT_VAL_BATCHES \
    --limit_test_batches $LIMIT_TEST_BATCHES \
    --cities "$CITY"

if [ $? -eq 0 ]; then
    log_and_echo " Training Success: San Antonio single-city experiment"
else
    log_and_echo "❌ Training Failed: San Antonio single-city experiment"
    exit 1
fi

log_and_echo ""
log_and_echo " Training completed\!"
log_and_echo " Check your WandB project '$WANDB_PROJECT' for results\!"
