# SE-AGCNet

## Dataset

The VoiceBankAGC dataset can be downloaded from:
[VoiceBankAGC](https://entuedu-my.sharepoint.com/:f:/r/personal/ccds-jmzhang_assoc_main_ntu_edu_sg/Documents/VoiceBankAGC?csf=1&web=1&e=gDmYcd)

## Quick Start

### Training

For training SE-AGCNet, please refer to:
```bash
./train.sh
```

Modify the data paths and training parameters in `train.sh` according to your setup.

### Inference

For inference with pre-trained models, please refer to:
```bash
./inference.sh
```

Two pre-trained models are included at `./SE_AGCNet/ckpt`.

### Data Generation

For details on SE-AGC data generation, please refer to the `DATAGEN/` directory.

## PyAGC

The `pyagc/` directory contains the Python 3 implementation of time-frequency automatic gain control. See `pyagc/README.md` for more details.



