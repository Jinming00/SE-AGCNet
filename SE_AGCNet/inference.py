"""Inference script for SE-AGCNet."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import os
import torch
import librosa
import soundfile as sf
import argparse
import json
import numpy as np
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.agc import MPSENetAGC


def load_model(checkpoint_path, config_path, device):
    """Load trained MP-SENet + AGC model."""
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    model = MPSENetAGC(h).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    
    return model, h


def normalize_audio_rms(audio):
    """RMS normalization consistent with training."""
    norm_factor = torch.sqrt(len(audio) / torch.sum(audio ** 2.0))
    normalized_audio = audio * norm_factor
    return normalized_audio, norm_factor


def denormalize_audio_rms(normalized_audio, norm_factor):
    """Reverse RMS normalization."""
    return normalized_audio / norm_factor


def split_audio(audio, segment_length, overlap_ratio=0.5):
    """
    Split audio into overlapping segments.
    Returns: (segments, segment_info)
    """
    segments = []
    segment_info = []
    overlap_length = int(segment_length * overlap_ratio)
    step_size = segment_length - overlap_length

    start = 0
    while start < len(audio):
        end = min(start + segment_length, len(audio))
        segment = audio[start:end]
        actual_length = len(segment)
        overlap_start = overlap_length if start > 0 else 0
        overlap_end = overlap_length if end < len(audio) else 0
        
        if len(segment) < segment_length:
            segment = np.pad(segment, (0, segment_length - len(segment)))

        segments.append(segment)
        segment_info.append((start, actual_length, overlap_start, overlap_end))

       
        if end >= len(audio):
            break

        start += step_size

    return segments, segment_info

def merge_audio_segments(segments, segment_info, original_length):
    """Merge overlapping audio segments by averaging."""
    merged = np.zeros(original_length)
    count = np.zeros(original_length)

    for segment, (start, actual_length, overlap_start, overlap_end) in zip(segments, segment_info):
        end = start + actual_length
        segment_data = segment[:actual_length]
        merged[start:end] += segment_data
        count[start:end] += 1

    count[count == 0] = 1
    merged = merged / count
    return merged

def process_segments_batch(model, h, segments, device, batch_size=16, use_chunk_norm=True, global_norm_factor=None):
    """Process audio segments in batches with peak normalization."""
    enhanced_segments = []

    for i in range(0, len(segments), batch_size):
        batch_segments = segments[i:i+batch_size]
        batch_tensors = []
        batch_norm_factors = []

        for segment in batch_segments:
            segment_tensor = torch.FloatTensor(segment).to(device)
            
            if use_chunk_norm:
                norm_factor = torch.sqrt(len(segment_tensor) / torch.sum(segment_tensor ** 2.0)).to(device)
            else:
                norm_factor = torch.tensor(global_norm_factor).to(device) if global_norm_factor is not None else torch.sqrt(len(segment_tensor) / torch.sum(segment_tensor ** 2.0)).to(device)

            segment_tensor = segment_tensor * norm_factor
            batch_tensors.append(segment_tensor)
            batch_norm_factors.append(norm_factor)

        batch_tensor = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            torch.cuda.empty_cache()
            noisy_mag, noisy_pha, _ = mag_pha_stft(batch_tensor, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            norm_factor_tensor = torch.stack(batch_norm_factors).to(device)
            agc_mag_normalized, mpnet_pha, _, _, _, agc_norm_factor = model(noisy_mag, noisy_pha, norm_factor_tensor)
            audio_g = mag_pha_istft(agc_mag_normalized, mpnet_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            for j in range(audio_g.shape[0]):
                enhanced_segment = audio_g[j].cpu().numpy()
                peak_value = np.max(np.abs(enhanced_segment))
                if peak_value > 0:
                    enhanced_segment = enhanced_segment * (0.4 / peak_value)
                enhanced_segments.append(enhanced_segment)

    return enhanced_segments


def inference_single_file(model, h, input_file, output_file, device, max_length=32000, batch_size=16,
                         use_chunk_norm=True, overlap_ratio=0.5):
    """Process single audio file with optional chunking and overlap."""
    try:
        noisy_wav, sr = sf.read(input_file)

        if sr != h.sampling_rate:
            import scipy.signal
            noisy_wav = scipy.signal.resample(noisy_wav, int(len(noisy_wav) * h.sampling_rate / sr))

        original_length = len(noisy_wav)

        if len(noisy_wav) <= max_length:
            norm_factor = np.sqrt(len(noisy_wav) / np.sum(noisy_wav ** 2.0))
            noisy_wav_norm = noisy_wav * norm_factor

            noisy_wav_tensor = torch.FloatTensor(noisy_wav_norm).unsqueeze(0).to(device)
            norm_factor_tensor = torch.FloatTensor([norm_factor]).to(device)

            with torch.no_grad():
                noisy_mag, noisy_pha, _ = mag_pha_stft(noisy_wav_tensor, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                agc_mag_normalized, mpnet_pha, _, _, _, agc_norm_factor = model(noisy_mag, noisy_pha, norm_factor_tensor)
                audio_g = mag_pha_istft(agc_mag_normalized, mpnet_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                enhanced_audio = audio_g.squeeze().cpu().numpy()
                peak_value = np.max(np.abs(enhanced_audio))
                if peak_value > 0:
                    enhanced_audio = enhanced_audio * (0.4 / peak_value)
        else:
            if use_chunk_norm:
                segments, segment_info = split_audio(noisy_wav, max_length, overlap_ratio)
                enhanced_segments = process_segments_batch(model, h, segments, device, batch_size, use_chunk_norm=True)
            else:
                norm_factor = np.sqrt(len(noisy_wav) / np.sum(noisy_wav ** 2.0))
                segments, segment_info = split_audio(noisy_wav, max_length, overlap_ratio)
                enhanced_segments = process_segments_batch(model, h, segments, device, batch_size,
                                                         use_chunk_norm=False, global_norm_factor=norm_factor)
            enhanced_audio = merge_audio_segments(enhanced_segments, segment_info, original_length)

        sf.write(output_file, enhanced_audio, h.sampling_rate, 'PCM_16')
        return True

    except Exception as e:
        print(f"Error processing {os.path.basename(input_file)}: {str(e)}")
        return False


def inference(input_dir, output_dir, model, h, device, max_length=32000, batch_size=16,
              use_chunk_norm=True, overlap_ratio=0.5):
    """Process directory of audio files."""
    audio_extensions = ('.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3', '.m4a', '.M4A')

    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    all_files = os.listdir(input_dir)
    test_indexes = [f for f in all_files
                   if os.path.isfile(os.path.join(input_dir, f)) and
                   f.endswith(audio_extensions)]

    print(f"Directory: {input_dir}")
    print(f"Total files in directory: {len(all_files)}")
    print(f"Audio files found: {len(test_indexes)}")

    if len(test_indexes) == 0:
        print("No audio files found! Checking file extensions...")
        print("Files in directory:", all_files[:10])
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Using segment length: {max_length/h.sampling_rate:.1f}s")
    print(f"Using batch size: {batch_size}")
    print(f"Using chunk norm: {'Yes' if use_chunk_norm else 'No (global norm)'}")
    print(f"Using overlap ratio: {overlap_ratio:.1%}")
    print(f"Output directory: {output_dir}")

    successful_count = 0
    failed_count = 0

    for index in test_indexes:
        input_file = os.path.join(input_dir, index)
        output_file = os.path.join(output_dir, index)

        print(f"Processing: {index}")

        if inference_single_file(model, h, input_file, output_file, device, max_length, batch_size,
                                use_chunk_norm, overlap_ratio):
            successful_count += 1
            print(f"✓ Successfully processed: {index}")
        else:
            failed_count += 1
            print(f"✗ Failed to process: {index}")

    print(f"\nInference completed!")
    print(f"Successfully processed: {successful_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {output_dir}")

    return successful_count, failed_count


def main():
    parser = argparse.ArgumentParser(description='MP-SENet + AGC Audio Enhancement Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--input', required=True, help='Input audio directory')
    parser.add_argument('--output', required=True, help='Output audio directory')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--max_length', default=32000, type=int, help='Maximum segment length (samples)')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for segment processing')
    parser.add_argument('--use_chunk_norm', action='store_true', default=True,
                       help='Use chunk normalization (default: True)')
    parser.add_argument('--use_global_norm', action='store_true',
                       help='Use global normalization instead of chunk normalization')
    parser.add_argument('--overlap_ratio', default=0.5, type=float,
                       help='Overlap ratio for segment processing (0.0-0.5, default: 0.5)')

    args = parser.parse_args()

    use_chunk_norm = args.use_chunk_norm and not args.use_global_norm
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model, h = load_model(args.checkpoint, args.config, device)
    print("Model loaded successfully!")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")

    if os.path.isfile(args.input):
        output_file = args.output
        if inference_single_file(model, h, args.input, output_file, device, args.max_length, args.batch_size,
                                use_chunk_norm, args.overlap_ratio):
            print(f"Successfully processed: {args.input}")
        else:
            print(f"Failed to process: {args.input}")
    elif os.path.isdir(args.input):
        inference(args.input, args.output, model, h, device, args.max_length, args.batch_size,
                 use_chunk_norm, args.overlap_ratio)
    else:
        print(f"Error: Input path {args.input} does not exist!")
        return

    print("All processing completed!")


if __name__ == '__main__':
    main()


