#!/bin/bash
# SE-AGCNet Inference Script

# cd /home/ccds-jmzhang/SE-AGCNet

# Required paths (modify these)
CHECKPOINT="./SE_AGCNet/ckpt/LibriAGC/g_00102000"
CONFIG="./SE_AGCNet/config.json"
INPUT_PATH="/home/ccds-jmzhang/10samples/noisy"       
OUTPUT_PATH="/home/ccds-jmzhang/10samples/enhanced"   

# Optional parameters
MAX_LENGTH=32000  # Segment length in samples (2s at 16kHz)
BATCH_SIZE=16     

# Check if paths are set
if [ "$INPUT_PATH" = "/path/to/input" ] || [ "$OUTPUT_PATH" = "/path/to/output" ]; then
    echo "Error: Please set INPUT_PATH and OUTPUT_PATH!"
    echo ""
    echo "Examples:"
    echo "  Single file: INPUT_PATH=\"/path/to/noisy.wav\" OUTPUT_PATH=\"/path/to/enhanced.wav\""
    echo "  Directory:   INPUT_PATH=\"/path/to/noisy/dir\" OUTPUT_PATH=\"/path/to/output/dir\""
    exit 1
fi

# Run inference
echo "Starting inference..."
echo "Input:  $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo ""

python SE_AGCNet/inference.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --input $INPUT_PATH \
    --output $OUTPUT_PATH \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE

echo ""
echo "Inference completed!"
