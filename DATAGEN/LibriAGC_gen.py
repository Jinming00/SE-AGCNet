"""
LibriAGC Dataset Generator

Processes LibriTTS (train-clean-100/360, test-clean) to generate 
multi-speaker audio combinations for AGC simulation.

Combines 2-5 clips from different speakers into:
- origin/: Concatenated audio
- lower/: Volume-reduced (5-30%) + augmented audio
- transcriptions/: Text files
- rttm/: Speaker diarization
- metadata/: Processing info
"""

import os
import numpy as np
import librosa
import soundfile as sf
import random
import glob
import sys
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global random seed for reproducibility
GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Configure data paths
DATA_BASE_DIR = os.environ.get('LIBRITTS_BASE_DIR', '/home/users/ntu/ccdsjmzh/scratch')


def batch_process_libritts_dataset(dataset_type='train'):
    """
    Scan LibriTTS dataset and group audio files by speaker.
    
    Args:
        dataset_type: 'train' for train-clean-100/360, 'test' for test-clean
    
    Returns:
        tuple: (speaker_files dict, source_dir path, target_dir path, target_sr)
    """
    if dataset_type == 'train':
        source_dirs = [
        os.path.join(DATA_BASE_DIR, "LibriTTS", "train-clean-100"),
            # os.path.join(DATA_BASE_DIR, "LibriTTS", "train-clean-360")
        ]
        target_dir = os.path.join(DATA_BASE_DIR, "LibriTTS", "train_5_30")
    else:  # test
        source_dirs = [os.path.join(DATA_BASE_DIR, "LibriTTS", "test-clean")]
        target_dir = os.path.join(DATA_BASE_DIR, "LibriTTS", "test_5_30")
    
    target_sr = 16000

    print(f"\nProcessing LibriTTS ({dataset_type})")
    print(f"Target: {target_dir}")
    print(f"Sample rate: {target_sr} Hz")

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "origin"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "lower"), exist_ok=True)

    speaker_files = defaultdict(list)
    total_files = 0

    print(f"\nScanning directories...")
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"  Skipping: {source_dir}")
            continue

        print(f"  {os.path.basename(source_dir)}: ", end="")
        wav_files = glob.glob(os.path.join(source_dir, "**/*.wav"), recursive=True)
        print(f"{len(wav_files)} files")
        
        for wav_file in tqdm(wav_files, desc=f"     Processing {os.path.basename(source_dir)}", 
                             unit="file", ncols=100):
            filename = os.path.basename(wav_file)
            speaker_id = filename.split('_')[0]

            try:
                duration = librosa.get_duration(path=wav_file)
                if duration > 0.5:
                    speaker_files[speaker_id].append(wav_file)
                    total_files += 1
            except Exception as e:
                # Skip corrupted files
                continue

    print(f"Valid files: {total_files}, Speakers: {len(speaker_files)}")
    
    existing_source_dir = next((d for d in source_dirs if os.path.exists(d)), None)
    return speaker_files, existing_source_dir, target_dir, target_sr


def create_audio_combinations(speaker_files, target_dir, target_sr, num_combinations=50):
    """
    Create maximum number of audio combinations using each file only once.
    Uses multi-threading for efficient processing.
    
    Args:
        speaker_files: Dictionary of files grouped by speaker
        target_dir: Output directory
        target_sr: Target sample rate
        num_combinations: Target number of combinations (not used, creates maximum possible)
        
    Returns:
        list: Created combinations
    """
    print(f"\nCreating combinations...")
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
    
    combination_configs = []
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
        
        # Ensure at least 2 different speakers
        for speaker in shuffled_speakers[:min(2, len(shuffled_speakers))]:
            speaker_files_available = [f for f in available_files if f['speaker'] == speaker and not f['used']]
            if speaker_files_available:
                selected_file = random.choice(speaker_files_available)
                selected_files_info.append(selected_file)
                if len(selected_files_info) >= num_files:
                    break
        
        # Add more files if needed
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
        
        # Mark files as used
        for file_info in selected_files_info:
            file_info['used'] = True
        
        selected_files = [f['file'] for f in selected_files_info]
        
        # Create volume settings: at least one original, at least one reduced (5-30%)
        volume_settings = []
        for i in range(len(selected_files)):
            if random.random() < 0.33:
                volume_settings.append(1.0)
            else:
                volume_percent = random.uniform(5, 30)
                volume_settings.append(volume_percent / 100.0)
        
        # Ensure at least one original and one reduced volume
        has_original = any(abs(v - 1.0) < 0.001 for v in volume_settings)
        has_reduced = any(v < 0.99 for v in volume_settings)
        
        if not has_original:
            random_idx = random.randint(0, len(volume_settings) - 1)
            volume_settings[random_idx] = 1.0
        
        if not has_reduced:
            random_idx = random.randint(0, len(volume_settings) - 1)
            volume_settings[random_idx] = random.uniform(5, 30) / 100.0
        
        combo_idx += 1
        combination_configs.append({
            'combo_id': combo_idx,
            'file_paths': selected_files,
            'volume_settings': volume_settings
        })
        
    # Multi-threading processing
    print(f"Generated {len(combination_configs)} configs, processing...")
    
    combinations_created = []
    lock = threading.Lock()
    
    def process_combination_wrapper(config):
        try:
            thread_seed = GLOBAL_RANDOM_SEED + config['combo_id']
            random.seed(thread_seed)
            np.random.seed(thread_seed)
            
            origin_result = process_audio_combination_origin(
                config['file_paths'], target_dir, target_sr, config['combo_id']
            )
            
            lower_result = process_audio_combination_lower(
                config['file_paths'], config['volume_settings'], 
                target_dir, target_sr, config['combo_id']
            )
            
            if origin_result and lower_result:
                combined_result = origin_result.copy()
                combined_result['lower_output_file'] = lower_result['lower_output_file']
                combined_result['file_info'] = lower_result['file_info']
                combined_result['file_info_origin'] = origin_result['file_info']
                return combined_result
            else:
                return origin_result or lower_result
                
        except Exception as e:
            return None
    
    max_workers = 8
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_config = {executor.submit(process_combination_wrapper, config): config 
                           for config in combination_configs}
        
        for future in tqdm(as_completed(future_to_config), total=len(combination_configs), 
                          desc="Processing", unit="combo", ncols=80):
            result = future.result()
            if result:
                with lock:
                    combinations_created.append(result)
    
    print(f"Created {len(combinations_created)} combinations")
    return combinations_created


def process_audio_combination_origin(file_paths, target_dir, target_sr, combo_id):
    """
    Process origin audio combination - simple concatenation without augmentation.
    
    Args:
        file_paths: List of audio file paths
        target_dir: Output directory
        target_sr: Target sample rate
        combo_id: Combination ID
        
    Returns:
        dict: Combination information
    """
    try:
        audio_segments = []
        file_info = []

        for file_path in file_paths:
            audio, sr = librosa.load(file_path, sr=target_sr)
            audio_segments.append(audio.copy())

            filename = os.path.basename(file_path)
            speaker_id = filename.split('_')[0]

            file_info.append({
                'original_file': file_path,
                'speaker_id': speaker_id,
                'volume': 1.0,
                'duration': len(audio) / target_sr,
                'augmentation_mode': 'none',
                'augmentation_description': 'No augmentation - simple concatenation'
            })

        combined_audio = np.concatenate(audio_segments)
        total_duration = len(combined_audio) / target_sr

        speakers = list(set([info['speaker_id'] for info in file_info]))
        speakers_str = '_'.join(sorted(speakers))
        base_filename = f"combo_{combo_id:04d}_speakers_{speakers_str}_{len(file_paths)}files_{total_duration:.1f}s.wav"

        origin_output_path = os.path.join(target_dir, "origin", base_filename)
        sf.write(origin_output_path, combined_audio, target_sr)

        return {
            'combo_id': combo_id,
            'origin_output_file': origin_output_path,
            'speakers': speakers,
            'total_duration': total_duration,
            'num_segments': len(file_paths),
            'file_info': file_info
        }

    except Exception as e:
        return None


def process_audio_combination_lower(file_paths, volume_settings, target_dir, target_sr, combo_id):
    """
    Process lower audio combination with volume adjustment and augmentation.
    
    Args:
        file_paths: List of audio file paths
        volume_settings: List of volume factors (0.05-0.30 or 1.0)
        target_dir: Output directory
        target_sr: Target sample rate
        combo_id: Combination ID
        
    Returns:
        dict: Combination information
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.append(project_root)

        from audio_augmentation import AudioAugmentation
        
        augmenter = AudioAugmentation(random_seed=42 + combo_id)
        lower_audio_segments = []
        file_info = []

        for file_path, volume in zip(file_paths, volume_settings):
            audio, sr = librosa.load(file_path, sr=target_sr)
            
            # Apply augmentation (15% probability per mode, 60% total)
            augmented_audio, applied_mode = augmenter.apply_augmentation(
                audio, target_sr, probability=0.15
            )
            
            audio_adjusted = augmented_audio * volume
            lower_audio_segments.append(audio_adjusted)

            filename = os.path.basename(file_path)
            speaker_id = filename.split('_')[0]

            file_info.append({
                'original_file': file_path,
                'speaker_id': speaker_id,
                'volume': volume,
                'duration': len(augmented_audio) / target_sr,
                'augmentation_mode': applied_mode,
                'augmentation_description': augmenter.get_mode_description(applied_mode)
            })

        lower_combined_audio = np.concatenate(lower_audio_segments)
        total_duration = len(lower_combined_audio) / target_sr

        speakers = list(set([info['speaker_id'] for info in file_info]))
        speakers_str = '_'.join(sorted(speakers))
        base_filename = f"combo_{combo_id:04d}_speakers_{speakers_str}_{len(file_paths)}files_{total_duration:.1f}s.wav"

        lower_output_path = os.path.join(target_dir, "lower", base_filename)
        sf.write(lower_output_path, lower_combined_audio, target_sr)

        return {
            'combo_id': combo_id,
            'lower_output_file': lower_output_path,
            'speakers': speakers,
            'total_duration': total_duration,
            'num_segments': len(file_paths),
            'file_info': file_info
        }

    except Exception as e:
        return None


def generate_transcription_files(combinations, source_dir, target_dir):
    """
    Generate transcription and RTTM files for audio combinations.
    
    Args:
        combinations: List of audio combinations
        source_dir: LibriTTS source directory
        target_dir: Output directory
        
    Returns:
        tuple: (transcription_dir, rttm_dir)
    """
    print(f"\nGenerating transcriptions...")
    transcription_dir = os.path.join(target_dir, "transcriptions")
    rttm_dir = os.path.join(target_dir, "rttm")
    
    for sub_dir in [transcription_dir, rttm_dir]:
        os.makedirs(sub_dir, exist_ok=True)
    
    for combo in tqdm(combinations, desc="TXT/RTTM", unit="file", ncols=80):
        try:
            combo_id = combo['combo_id']
            
            if 'origin_output_file' in combo:
                output_filename = os.path.basename(combo['origin_output_file']).replace('.wav', '')
            elif 'output_file' in combo:
                output_filename = os.path.basename(combo['output_file']).replace('.wav', '')
            else:
                speakers = '_'.join(sorted(combo['speakers']))
                output_filename = f"combo_{combo_id:04d}_speakers_{speakers}"
            
            all_transcriptions = []
            segment_info = []
            current_time = 0.0
            
            for segment in combo['file_info']:
                original_file = segment['original_file']
                filename = os.path.basename(original_file).replace('.wav', '')
                path_parts = original_file.split('/')
                speaker_id = path_parts[-3]
                
                txt_filename = filename + '.normalized.txt'
                txt_path = os.path.join(os.path.dirname(original_file), txt_filename)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    all_transcriptions.append(text)
                    duration = segment['duration']
                    end_time = current_time + duration
                    
                    segment_info.append({
                        'speaker_id': segment['speaker_id'],
                        'start_time': current_time,
                        'end_time': end_time,
                        'duration': duration,
                        'text': text,
                        'volume': segment['volume']
                    })
                    
                    current_time = end_time
                else:
                    segment_info.append({
                        'speaker_id': segment['speaker_id'],
                        'start_time': current_time,
                        'end_time': current_time + segment['duration'],
                        'duration': segment['duration'],
                        'text': '[MISSING_TEXT]',
                        'volume': segment['volume']
                    })
                    current_time += segment['duration']
            
            # Generate transcription file
            transcription_file = os.path.join(transcription_dir, f"{output_filename}.txt")
            with open(transcription_file, 'w', encoding='utf-8') as f:
                combined_text = ' '.join(all_transcriptions)
                f.write(combined_text)
            
            # Generate RTTM file
            rttm_file = os.path.join(rttm_dir, f"{output_filename}.rttm")
            with open(rttm_file, 'w', encoding='utf-8') as f:
                for segment in segment_info:
                    f.write(f"SPEAKER {output_filename} 1 {segment['start_time']:.3f} {segment['duration']:.3f} <NA> <NA> {segment['speaker_id']} <NA> <NA>\n")
            
        except Exception as e:
            continue
    
    return transcription_dir, rttm_dir


def main_train():
    """Process training datasets (train-clean-100 and train-clean-360)."""
    speaker_files, source_dir, target_dir, target_sr = batch_process_libritts_dataset('train')
    
    if not speaker_files:
        print("\nNo valid speaker files found.")
        return
    
    combinations = create_audio_combinations(speaker_files, target_dir, target_sr, 50)
    
    if combinations:
        transcription_dir, rttm_dir = generate_transcription_files(
            combinations, source_dir, target_dir
        )
        
        print(f"\nGenerating metadata...")
        from metadata_generator import generate_metadata_files
        metadata_result = generate_metadata_files(combinations, target_dir)
        
        print(f"\nProcessing complete.")
        print(f"Combinations: {len(combinations)}")
        print(f"Output: {target_dir}\n")


def main_test():
    """Process test dataset (test-clean)."""
    speaker_files, source_dir, target_dir, target_sr = batch_process_libritts_dataset('test')
    
    if not speaker_files:
        print("\nNo valid speaker files found.")
        return
    
    combinations = create_audio_combinations(speaker_files, target_dir, target_sr, 50)
    
    if combinations:
        transcription_dir, rttm_dir = generate_transcription_files(
            combinations, source_dir, target_dir
        )
        
        print(f"\nGenerating metadata...")
        from metadata_generator import generate_metadata_files
        metadata_result = generate_metadata_files(combinations, target_dir)
        
        print(f"\nProcessing complete.")
        print(f"Combinations: {len(combinations)}")
        print(f"Output: {target_dir}\n")


if __name__ == "__main__":
    # Uncomment the one you want to process
    # main_train()  # Process training datasets
    main_test()     # Process test dataset
