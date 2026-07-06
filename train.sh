#!/bin/bash
# SE-AGCNet Training Script
#PBS -N seagcnet
#PBS -l select=1:ngpus=2:ncpus=16
#PBS -l walltime=240:00:00
#PBS -q gpu_a40
#PBS -P scse_aseschng 

cd /home/ccds-jmzhang/SE-AGCNet/
source ~/.bashrc    
conda activate seagcnet

# export PYTHONNOUSERSITE=1
# unset PYTHONPATH
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cufft/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

# Data paths (modify these)
TRAIN_NOISY_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/train_5_30/lower_noisy"    # noisy and volume-unbalanced audio (input)
TRAIN_CLEAN_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/train_5_30/lower"    # clean and volume-unbalanced audio (SE target)
TRAIN_ORIGIN_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/train_5_30/origin"  # clean and volume-balanced audio (AGC target)

CHECKPOINT_DIR="/projects_vol/gp_aseschng/jinming/ckpts/seagcnet/test"  # where to save checkpoints and logs
CONFIG_FILE="/home/ccds-jmzhang/SE-AGCNet/SE_AGCNet/config.json"

# EXTRA_VALIDATION_CLEAN_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/test_5_30/origin"
# EXTRA_VALIDATION_NOISY_DIR="/projects_vol/gp_aseschng/jinming/data/LibriAGC/test_5_30/lower_noisy"

# Weights & Biases monitoring
# Make sure you have run `wandb login` before online logging.
USE_WANDB=true
WANDB_PROJECT="SE-AGCNet"
WANDB_ENTITY=""
WANDB_RUN_NAME="se-agcnet-$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="online"      # online, offline, or disabled
WANDB_WATCH="gradients"  # gradients, parameters, all, or false

# DNSMOS validation
VALIDATION_DNSMOS_ENABLED=1
VALIDATION_DNSMOS_PATH="/home/ccds-jmzhang/SE-AGCNet/SE_AGCNet/assets/dnsmos"

# Run training
TRAIN_ARGS=(
    --input_train_clean_dir "$TRAIN_CLEAN_DIR"
    --input_train_noisy_dir "$TRAIN_NOISY_DIR"
    --input_train_origin_dir "$TRAIN_ORIGIN_DIR"
    --input_test_clean_dir "$EXTRA_VALIDATION_CLEAN_DIR"
    --input_test_noisy_dir "$EXTRA_VALIDATION_NOISY_DIR"
    --checkpoint_path "$CHECKPOINT_DIR"
    --config "$CONFIG_FILE"
    --validation_dnsmos_enabled "$VALIDATION_DNSMOS_ENABLED"
    --validation_dnsmos_path "$VALIDATION_DNSMOS_PATH"
)

if [ "$USE_WANDB" = true ]; then
    TRAIN_ARGS+=(--use_wandb)
    TRAIN_ARGS+=(--wandb_project "$WANDB_PROJECT")
    TRAIN_ARGS+=(--wandb_run_name "$WANDB_RUN_NAME")
    TRAIN_ARGS+=(--wandb_mode "$WANDB_MODE")
    TRAIN_ARGS+=(--wandb_watch "$WANDB_WATCH")

    if [ -n "$WANDB_ENTITY" ]; then
        TRAIN_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
fi

python SE_AGCNet/train.py "${TRAIN_ARGS[@]}"

echo "Training completed!"
