#!/bin/bash

cd /home/ccds-jmzhang/SE-AGCNet
source ~/.bashrc
set -euo pipefail
conda activate seagcnet

INPUT_PATH="/projects_vol/gp_aseschng/jinming/data/LibriAGC/test_5_30/lower"
OUTPUT_PATH="/projects_vol/gp_aseschng/jinming/data/LibriAGC/test_5_30/scratch"
MODEL_PATH="/home/ccds-jmzhang/SE-AGCNet/single_agc/model/agc_model_best.pt"

if [ ! -e "$INPUT_PATH" ]; then
    echo "Error: INPUT_PATH does not exist: $INPUT_PATH"
    exit 1
fi

if [ -d "$INPUT_PATH" ]; then
    mkdir -p "$OUTPUT_PATH"
    shopt -s nullglob
    inputs=("$INPUT_PATH"/*.wav)
    shopt -u nullglob

    if [ ${#inputs[@]} -eq 0 ]; then
        echo "Error: no .wav files found in $INPUT_PATH"
        exit 1
    fi
else
    inputs=("$INPUT_PATH")
fi

for input_file in "${inputs[@]}"; do
    if [ -d "$INPUT_PATH" ]; then
        output_file="$OUTPUT_PATH/$(basename "$input_file")"
        echo "Processing $(basename "$input_file")"
    else
        output_file="$OUTPUT_PATH"
    fi

    python single_agc/inference.py \
        "$input_file" \
        "$output_file" \
        "$MODEL_PATH"
done
