#!/bin/bash

# Set your dataset path 
DATA_DIR="/scratch/ccds-jmzhang/data/LibriTTS"

# Output directory
OUTPUT_DIR="/scratch/ccds-jmzhang/data/LibriAGC"

# Processing mode: test or train
MODE="train"

python /home/ccds-jmzhang/SE-AGCNet/DATAGEN/LibriAGC_gen.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --mode ${MODE}
