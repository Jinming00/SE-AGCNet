# AGC Dataset Generation

Generate multi-speaker audio combinations for AGC speech enhancement training.

## Datasets

- **LibriTTS**: https://www.openslr.org/60/
- **VoiceBank-Demand(use '28spk')**: https://datashare.ed.ac.uk/handle/10283/2791  

## Requirements

```bash
pip install numpy librosa soundfile tqdm pandas
```

## Usage

### LibriTTS

```bash
python LibriAGC_gen.py \
    --data_dir /path/to/LibriTTS \
    --output_dir /path/to/output \
    --mode test  # or train
```

Or use the script:
```bash
bash run_libriagc_gen.sh
```

### VoiceBank-Demand

```bash
python VoiceBankAGC_gen.py \
    --data_dir /path/to/voicebank-demand \
    --output_dir /path/to/output
```

Or:
```bash
bash run_voicebank_gen.sh
```

## How it works

Concatenates 2-5 clips from different speakers:

| Directory | Content |
|-----------|---------|
| origin/ | Raw concatenation |
| lower/ | Some clips reduced to 5-30% volume, with augmentation |

Augmentation (15% each):
- Sudden spikes (2-5x volume)
- Gradual increase
- Gradual decrease
- Volume fluctuation (sine wave)

## Output

```
{output_dir}/{mode}_5_30/
├── origin/
├── lower/
├── transcriptions/   # LibriTTS only
├── rttm/             # LibriTTS only
└── metadata/
```


