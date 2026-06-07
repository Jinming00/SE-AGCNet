#!/bin/bash

# Input lower directory containing wav files.
LOWER_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/train_5_30/lower"

# Noise directory.
NOISE_DIR="/projects_vol/gp_aseschng/jinming/data/NOISE/DNS_for_SE-AGCNet"

# Output directory. Leave empty to use the default sibling directory: lower_noisy
OUTPUT_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/train_5_30/lower_noisy"

MIN_SNR=5
MAX_SNR=25
SAMPLE_RATE=16000
SEED=42
NUM_WORKERS=16

CMD=(python /home/ccds-jmzhang/SE-AGCNet/DATAGEN/add_noise_to_lower.py
    --lower_dir "${LOWER_DIR}"
    --noise_dir "${NOISE_DIR}"
    --min_snr "${MIN_SNR}"
    --max_snr "${MAX_SNR}"
    --sample_rate "${SAMPLE_RATE}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}")

if [ -n "${OUTPUT_DIR}" ]; then
    CMD+=(--output_dir "${OUTPUT_DIR}")
fi

"${CMD[@]}"
