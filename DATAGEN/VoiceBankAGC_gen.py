"""
VoiceBank-AGC Dataset Generator

Processes VoiceBank-Demand to generate multi-speaker audio combinations for AGC simulation.

Combines 2-5 clips from different speakers into:
- origin/: Concatenated audio (optional peak normalization)
- lower/: Volume-reduced (5-30%) audio
- metadata/: Processing info

Peak normalization disabled by default. Enable with:
main_process_voicebank_demand(enable_origin_peak_normalization=True, target_peak=0.4)
"""

import os
import numpy as np
import librosa
import soundfile as sf
import glob
from tqdm import tqdm
from collections import defaultdict
import random
import json
import pandas as pd

# Global random seed for reproducibility
GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Configure data paths
VOICEBANK_BASE_DIR = "/home/users/ntu/ccdsjmzh/scratch/voicebank-demand"
CLEAN_TRAINSET_DIR = os.path.join(VOICEBANK_BASE_DIR, "clean_trainset_wav")


def extract_speaker_from_filename(filename):
    """
    Extract speaker ID from VoiceBank-Demand filename.
    Example: p226_002.wav -> speaker_id = "226"
    
    Args:
        filename: Audio filename
        
    Returns:
        str: Speaker ID
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    if len(parts) >= 1 and parts[0].startswith('p'):
        return parts[0][1:]
    return basename


def scan_voicebank_dataset():
    """
    Scan VoiceBank-Demand clean_trainset_wav dataset.
    Groups audio files by speaker and filters valid files.
    
    Returns:
        tuple: (speaker_files dict, source_dir path)
    """
    print(f"\nProcessing VoiceBank-Demand")
    print(f"Source: {CLEAN_TRAINSET_DIR}")
    
    if not os.path.exists(CLEAN_TRAINSET_DIR):
        print(f"Error: Directory not found")
        return {}, None
    
    print(f"\nScanning directory...")
    wav_files = glob.glob(os.path.join(CLEAN_TRAINSET_DIR, "*.wav"))
    print(f"Found {len(wav_files)} files")
    
    speaker_files = defaultdict(list)
    total_files = 0
    
    for wav_file in tqdm(wav_files, desc="Processing", unit="file", ncols=80):
        filename = os.path.basename(wav_file)
        speaker_id = extract_speaker_from_filename(filename)
        
        try:
            duration = librosa.get_duration(path=wav_file)
            if duration > 0.5:
                speaker_files[speaker_id].append(wav_file)
                total_files += 1
        except Exception as e:
            continue
    
    print(f"Valid files: {total_files}, Speakers: {len(speaker_files)}")
    
    return dict(speaker_files), CLEAN_TRAINSET_DIR


def normalize_audio_peak(audio, target_peak=0.8):
    """Normalize audio to target peak amplitude."""
    current_peak = np.max(np.abs(audio))
    if current_peak == 0:
        return audio
    return audio * (target_peak / current_peak)


def create_voicebank_audio_combinations(speaker_files, target_dir, target_sr=16000, 
                                        enable_origin_peak_normalization=False, target_peak=0.8):
    """
    Create maximum number of VoiceBank audio combinations using each file only once.
    
    Args:
        speaker_files: Dictionary of files grouped by speaker
        target_dir: Output directory
        target_sr: Target sample rate
        enable_origin_peak_normalization: Whether to apply peak normalization to origin audio
        target_peak: Target peak value for normalization
        
    Returns:
        list: Created combinations
    """
    print(f"\nCreating combinations...")
    print(f"Target: {target_dir}")
    if enable_origin_peak_normalization:
        print(f"Peak normalization: {target_peak}")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "origin"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "lower"), exist_ok=True)

    random.seed(GLOBAL_RANDOM_SEED)
    np.random.seed(GLOBAL_RANDOM_SEED)

    valid_speakers = {k: v for k, v in speaker_files.items() if len(v) >= 1}
    speaker_ids = list(valid_speakers.keys())

    if len(speaker_ids) < 2:
        print("Error: Need at least 2 speakers")
        return []

    all_files = []
    for speaker_id, files in valid_speakers.items():
        for file_path in files:
            all_files.append({'file': file_path, 'speaker': speaker_id, 'used': False})

    total_files = len(all_files)
    
    combinations_created = []
    combo_idx = 0

    while True:
        available_files = [f for f in all_files if not f['used']]
        
        if len(available_files) < 2:
            break
        
        available_speakers = list(set([f['speaker'] for f in available_files]))
        if len(available_speakers) < 2:
            break
        
        max_files_for_combo = min(5, len(available_files))
        num_files = random.randint(2, max_files_for_combo)
        
        selected_files_info = []
        shuffled_speakers = available_speakers.copy()
        random.shuffle(shuffled_speakers)
        
        for speaker in shuffled_speakers[:min(2, len(shuffled_speakers))]:
            speaker_files_available = [f for f in available_files if f['speaker'] == speaker and not f['used']]
            if speaker_files_available:
                selected_file = random.choice(speaker_files_available)
                selected_files_info.append(selected_file)
                if len(selected_files_info) >= num_files:
                    break
        
        while len(selected_files_info) < num_files:
            remaining_files = [f for f in available_files if f not in selected_files_info and not f['used']]
            if not remaining_files:
                break
            selected_file = random.choice(remaining_files)
            selected_files_info.append(selected_file)
        
        if len(selected_files_info) < 2:
            break
        
        combo_speakers = set([f['speaker'] for f in selected_files_info])
        if len(combo_speakers) < 2:
            continue
        
        for file_info in selected_files_info:
            file_info['used'] = True
        
        selected_files = [f['file'] for f in selected_files_info]
        
        volume_settings = []
        for i in range(len(selected_files)):
            if random.random() < 0.4:
                volume_settings.append(1.0)
            else:
                volume_percent = random.uniform(5, 30)
                volume_settings.append(volume_percent / 100.0)
        
        has_original = any(abs(v - 1.0) < 0.001 for v in volume_settings)
        has_reduced = any(v < 0.99 for v in volume_settings)
        
        if not has_original:
            random_idx = random.randint(0, len(volume_settings) - 1)
            volume_settings[random_idx] = 1.0
        
        if not has_reduced:
            random_idx = random.randint(0, len(volume_settings) - 1)
            volume_settings[random_idx] = random.uniform(5, 30) / 100.0
        
        combo_idx += 1
        
        try:
            combination_result = process_voicebank_combination(
                selected_files, volume_settings, target_dir, target_sr, combo_idx, 
                enable_origin_peak_normalization, target_peak
            )
            
            if combination_result:
                combinations_created.append(combination_result)
                    
        except Exception as e:
            continue
    
    print(f"Created {len(combinations_created)} combinations")
    return combinations_created


def process_voicebank_combination(file_paths, volume_settings, target_dir, target_sr, combo_id, 
                                  enable_origin_peak_normalization=False, target_peak=0.8):
    """
    Process single VoiceBank audio combination.
    
    Args:
        file_paths: List of audio file paths
        volume_settings: List of volume factors
        target_dir: Output directory
        target_sr: Target sample rate
        combo_id: Combination ID
        enable_origin_peak_normalization: Whether to apply peak normalization
        target_peak: Target peak value
        
    Returns:
        dict: Combination information
    """
    try:
        origin_audio_segments = []
        lower_audio_segments = []
        file_info = []

        for file_path, volume in zip(file_paths, volume_settings):
            audio, sr = librosa.load(file_path, sr=target_sr)

            if enable_origin_peak_normalization:
                origin_audio = normalize_audio_peak(audio.copy(), target_peak=target_peak)
                augmentation_mode = 'peak_normalized'
                augmentation_description = f'Peak normalized to {target_peak} before combination'
            else:
                origin_audio = audio.copy()
                augmentation_mode = 'original'
                augmentation_description = 'No normalization applied'
            
            origin_audio_segments.append(origin_audio)

            audio_adjusted = audio * volume
            lower_audio_segments.append(audio_adjusted)

            filename = os.path.basename(file_path)
            speaker_id = extract_speaker_from_filename(filename)

            file_info.append({
                'original_file': file_path,
                'filename': filename,
                'speaker_id': speaker_id,
                'volume': volume,
                'duration': len(audio) / target_sr,
                'sample_rate': target_sr,
                'origin_augmentation_mode': augmentation_mode,
                'origin_augmentation_description': augmentation_description,
                'origin_peak_normalized': enable_origin_peak_normalization,
                'target_peak': target_peak if enable_origin_peak_normalization else None
            })

        origin_combined_audio = np.concatenate(origin_audio_segments)
        lower_combined_audio = np.concatenate(lower_audio_segments)

        total_duration = len(origin_combined_audio) / target_sr

        speakers = list(set([info['speaker_id'] for info in file_info]))
        speakers_str = '_'.join(sorted(speakers))
        base_filename = f"voicebank_combo_{combo_id:04d}_speakers_{speakers_str}_{len(file_paths)}files_{total_duration:.1f}s.wav"

        origin_output_path = os.path.join(target_dir, "origin", base_filename)
        sf.write(origin_output_path, origin_combined_audio, target_sr)

        lower_output_path = os.path.join(target_dir, "lower", base_filename)
        sf.write(lower_output_path, lower_combined_audio, target_sr)

        return {
            'combo_id': combo_id,
            'origin_output_file': origin_output_path,
            'lower_output_file': lower_output_path,
            'speakers': speakers,
            'total_duration': total_duration,
            'num_segments': len(file_paths),
            'file_info': file_info,
            'base_filename': base_filename,
            'origin_peak_normalized': enable_origin_peak_normalization,
            'target_peak': target_peak if enable_origin_peak_normalization else None
        }

    except Exception as e:
        return None


def generate_voicebank_metadata(combinations, target_dir):
    """
    Generate metadata files for VoiceBank audio combinations.
    
    Args:
        combinations: List of audio combinations
        target_dir: Output directory
        
    Returns:
        dict: Metadata statistics
    """
    print(f"\nGenerating metadata...")
    metadata_dir = os.path.join(target_dir, "metadata")
    combinations_metadata_dir = os.path.join(metadata_dir, "combinations")
    os.makedirs(combinations_metadata_dir, exist_ok=True)

    total_duration = 0
    total_segments = 0
    speaker_stats = defaultdict(int)

    for combo in tqdm(combinations, desc="Metadata", unit="file", ncols=80):
        try:
            combo_id = combo['combo_id']

            metadata = {
                'combo_id': combo_id,
                'base_filename': combo['base_filename'],
                'origin_file': combo['origin_output_file'],
                'lower_file': combo['lower_output_file'],
                'speakers': combo['speakers'],
                'num_speakers': len(combo['speakers']),
                'total_duration': combo['total_duration'],
                'num_segments': combo['num_segments'],
                'origin_peak_normalized': combo.get('origin_peak_normalized', False),
                'target_peak': combo.get('target_peak', None),
                'segments': []
            }

            current_time = 0.0
            for segment in combo['file_info']:
                segment_metadata = {
                    'original_file': segment['original_file'],
                    'filename': segment['filename'],
                    'speaker_id': segment['speaker_id'],
                    'start_time': current_time,
                    'end_time': current_time + segment['duration'],
                    'duration': segment['duration'],
                    'volume_factor': segment['volume'],
                    'volume_percentage': int(segment['volume'] * 100),
                    'sample_rate': segment['sample_rate'],
                    'origin_augmentation_mode': segment.get('origin_augmentation_mode', 'original'),
                    'origin_augmentation_description': segment.get('origin_augmentation_description', 'No normalization applied'),
                    'origin_peak_normalized': segment.get('origin_peak_normalized', False),
                    'target_peak': segment.get('target_peak', None)
                }

                metadata['segments'].append(segment_metadata)
                current_time += segment['duration']

                speaker_stats[segment['speaker_id']] += 1

            metadata_file = os.path.join(combinations_metadata_dir, f"combo_{combo_id:04d}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            total_duration += combo['total_duration']
            total_segments += combo['num_segments']

        except Exception as e:
            continue

    overall_stats = {
        'dataset': 'voicebank-demand',
        'total_combinations': len(combinations),
        'total_segments': total_segments,
        'total_duration_seconds': total_duration,
        'total_duration_hours': total_duration / 3600,
        'unique_speakers': len(speaker_stats),
        'speaker_usage_stats': dict(speaker_stats),
        'generation_timestamp': pd.Timestamp.now().isoformat()
    }

    stats_file = os.path.join(metadata_dir, "overall_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)

    speaker_df = pd.DataFrame([
        {'speaker_id': speaker_id, 'usage_count': count}
        for speaker_id, count in speaker_stats.items()
    ]).sort_values('usage_count', ascending=False)

    speaker_stats_file = os.path.join(metadata_dir, "speaker_stats.csv")
    speaker_df.to_csv(speaker_stats_file, index=False)

    return {
        'total_combinations': len(combinations),
        'total_segments': total_segments,
        'total_duration_hours': total_duration / 3600,
        'metadata_dir': metadata_dir,
        'unique_speakers': len(speaker_stats)
    }


def main_process_voicebank_demand(enable_origin_peak_normalization=False, target_peak=0.4):
    """
    Main function to process VoiceBank-Demand dataset.
    
    Args:
        enable_origin_peak_normalization: Whether to apply peak normalization to origin audio
        target_peak: Target peak value for normalization
    """
    target_dir = os.path.join(VOICEBANK_BASE_DIR, "train_new_5_20")
    target_sr = 16000

    speaker_files, source_dir = scan_voicebank_dataset()

    if not speaker_files:
        print("\nNo valid speaker files found.")
        return

    combinations = create_voicebank_audio_combinations(
        speaker_files, target_dir, target_sr, enable_origin_peak_normalization, target_peak
    )

    if not combinations:
        print("\nNo combinations created.")
        return

    metadata_result = generate_voicebank_metadata(combinations, target_dir)

    print(f"\nProcessing complete.")
    print(f"Combinations: {len(combinations)}")
    print(f"Duration: {metadata_result['total_duration_hours']:.2f}h")
    print(f"Speakers: {metadata_result['unique_speakers']}")
    print(f"Output: {target_dir}\n")

    return combinations, metadata_result


if __name__ == "__main__":
    combinations, metadata_result = main_process_voicebank_demand(
        enable_origin_peak_normalization=False
    )
