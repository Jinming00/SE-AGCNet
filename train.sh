#!/bin/bash
# SE-AGCNet Training Script

# cd /home/ccds-jmzhang/SE-AGCNet


# Data paths (modify these)
TRAIN_NOISY_DIR="/home/ccds-jmzhang/LibriAGC/train/lower_noisy"    # noisy and volume-unbalanced audio (input)
TRAIN_CLEAN_DIR="/home/ccds-jmzhang/LibriAGC/train/lower_clean"    # clean and volume-unbalanced audio (SE target)
TRAIN_ORIGIN_DIR="/home/ccds-jmzhang/LibriAGC/train/origin_noisy"  # clean and volume-balanced audio (AGC target)

TEST_CLEAN_DIR="/home/ccds-jmzhang/LibriAGC/validate/lower_clean"
TEST_NOISY_DIR="/home/ccds-jmzhang/LibriAGC/validate/lower_noisy"

CHECKPOINT_DIR="/home/ccds-jmzhang/test"
CONFIG_FILE="./SE_AGCNet/config.json"

# Training parameters
TRAINING_EPOCHS=400
STDOUT_INTERVAL=10
CHECKPOINT_INTERVAL=1000
VALIDATION_INTERVAL=1000
BEST_CHECKPOINT_START_EPOCH=10

# Run training
python SE_AGCNet/train.py \
    --input_train_clean_dir "$TRAIN_CLEAN_DIR" \
    --input_train_noisy_dir "$TRAIN_NOISY_DIR" \
    --input_train_origin_dir "$TRAIN_ORIGIN_DIR" \
    --input_test_clean_dir "$TEST_CLEAN_DIR" \
    --input_test_noisy_dir "$TEST_NOISY_DIR" \
    --checkpoint_path "$CHECKPOINT_DIR" \
    --config "$CONFIG_FILE" \
    --training_epochs $TRAINING_EPOCHS \
    --stdout_interval $STDOUT_INTERVAL \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --validation_interval $VALIDATION_INTERVAL \
    --best_checkpoint_start_epoch $BEST_CHECKPOINT_START_EPOCH

echo "Training completed!"
