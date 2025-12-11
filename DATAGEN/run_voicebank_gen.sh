#!/bin/bash

# Set your dataset path (containing clean_trainset_wav/)
DATA_DIR="/scratch/ccds-jmzhang/data/voicebank-demand"

# Output directory
OUTPUT_DIR="/scratch/ccds-jmzhang/data/VoiceBankAGC"

python /home/ccds-jmzhang/SE-AGCNet/DATAGEN/VoiceBankAGC_gen.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR}
