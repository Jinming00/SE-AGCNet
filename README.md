# SE-AGCNet

SE-AGCNet is an end-to-end framework for joint speech enhancement (SE) and automatic gain control (AGC), designed for meeting scenarios with large loudness variation.

- Audio demos: https://jinming00.github.io/SE-AGCNet/

## Data Generation

For SE-AGC data generation details, see `DATAGEN/README.md`.

## Overview

Conventional audio pipelines often cascade SE and AGC as separate modules. Applying AGC before SE may amplify background noise, while applying SE before AGC may over-suppress quiet speech. SE-AGCNet jointly optimizes enhancement and loudness control so that quiet speech is preserved while output loudness remains consistent.

This repository includes:

- `SE_AGCNet/`: model, training, inference, metrics, and validation code.
- `single_agc/`: standalone AGC model training and inference code.
- `DATAGEN/`: SE-AGC data simulation scripts.
- `pyagc/`: Python implementation of time-frequency automatic gain control.
- `docs/`: static demo page assets with audio examples.

## Environment Setup

### Create the environment

```bash
conda env create -f environment.yml
conda activate seagcnet
```

## Quick Start

### Training

Update data paths and training parameters in `train.sh`, then run:

```bash
./train.sh
```

### Inference

Update checkpoint and input paths in `inference.sh`, then run:

```bash
./inference.sh
```

Pre-trained checkpoints are currently stored under `SE_AGCNet/ckpt/`.

### Standalone AGC

The standalone AGC model is under `single_agc/`.

For inference, update `INPUT_PATH`, `OUTPUT_PATH`, and `MODEL_PATH` in `single_agc/infer.sh`, then run:

```bash
bash single_agc/infer.sh
```

`INPUT_PATH` can be either a single `.wav` file or a directory containing `.wav` files.

## Runtime

Real-time factor (RTF) was measured on a single NVIDIA L40S GPU.

| Model | RTF |
| --- | ---: |
| MP-SENet | 0.0329 |
| SE-AGCNet | 0.0357 |

## PyAGC

The `pyagc/` directory contains the Python 3 implementation of time-frequency automatic gain control. See `pyagc/README.md` for details.

## Citation

Citation information will be added after publication.
