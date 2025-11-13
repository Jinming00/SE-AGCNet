# AGC Dataset Generation

Scripts for generating multi-speaker audio datasets with simulated AGC scenarios for speech enhancement.

## Supported Datasets

- **LibriTTS**: Multi-speaker English TTS corpus (train-clean-100, train-clean-360, test-clean)
- **VoiceBank-Demand**

## Quick Start

### Requirements

```bash
pip install numpy librosa soundfile tqdm pandas
```

### LibriTTS

Download the original dataset from https://www.openslr.org/60/

Set data directory:
```bash
export LIBRITTS_BASE_DIR=/path/to/your/data
```

Choose train or test in `LibriAGC_gen.py`:
```python
if __name__ == "__main__":
    main_train()  # or main_test()
```

Run:
```bash
python LibriAGC_gen.py
```

### VoiceBank-Demand

Edit path in `VoiceBankAGC_gen.py`:
```python
VOICEBANK_BASE_DIR = "/path/to/voicebank-demand"
```

Run:
```bash
python VoiceBankAGC_gen.py
```

## What It Does

Combines 2-5 audio clips from different speakers into single files. For each combination:

- **origin/**: Concatenated audio without modification
- **lower/**: Same audio with volume reduced to 5-30% of original

Audio augmentation is randomly applied to lower files (60% chance):
- Sudden volume spikes (15%)
- Gradual volume increase (15%)
- Gradual volume decrease (15%)
- Volume fluctuation (15%)

## Output Structure

### LibriTTS
```
LibriTTS/train_5_30/
├── origin/              # Concatenated audio
├── lower/               # Volume-reduced + augmented audio
├── transcriptions/      # Text transcriptions (.txt)
├── rttm/                # Speaker diarization (.rttm)
└── metadata/            # JSON metadata
```

### VoiceBank-Demand
```
voicebank-demand/processed/
├── origin/              # Concatenated audio
├── lower/               # Volume-reduced audio
└── metadata/            # JSON metadata
```


## Files

- `LibriAGC_gen.py`: LibriTTS dataset processor
- `VoiceBankAGC_gen.py`: VoiceBank dataset processor
- `audio_augmentation.py`: Augmentation functions
- `metadata_generator.py`: Metadata generation


