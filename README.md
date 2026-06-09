# SE-AGCNet

SE-AGCNet is an end-to-end framework for joint speech enhancement (SE) and automatic gain control (AGC), designed for meeting scenarios with large loudness variation.

- Project page and audio demos: https://jinming00.github.io/SE-AGCNet/
- Repository: https://github.com/Jinming00/SE-AGCNet

## Overview

Conventional audio pipelines often cascade SE and AGC as separate modules. Applying AGC before SE may amplify background noise, while applying SE before AGC may over-suppress quiet speech. SE-AGCNet jointly optimizes enhancement and loudness control so that quiet speech is preserved while output loudness remains consistent.

This repository includes:

- `SE_AGCNet/`: model, training, inference, metrics, and validation code.
- `DATAGEN/`: SE-AGC data simulation scripts.
- `pyagc/`: Python implementation of time-frequency automatic gain control.
- `docs/`: static demo page assets with audio examples.

## Dataset

The VoiceBankAGC dataset can be downloaded from:

[VoiceBankAGC](https://drive.google.com/drive/folders/12naNULZmxpUD7x06fQmLgCZXuH6-qxgD?usp=sharing)

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

### Data Generation

For SE-AGC data generation details, see `DATAGEN/README.md`.

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
