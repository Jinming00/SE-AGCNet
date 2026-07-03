import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import glob
from tqdm import tqdm
from pesq import pesq
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AGCModule(nn.Module):
    """
    AGC module operating on RMS-normalized linear magnitudes.
    """
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2, bidirectional=True, freq_bins=201):
        super(AGCModule, self).__init__()

        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.freq_bins = freq_bins

        self.freq_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=16 * freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * freq_bins)
        )

        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, input_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )

    def forward(self, magnitude):
        """
        Args:
            magnitude: [B, F, T] RMS-normalized magnitude
        """
        x = magnitude.unsqueeze(1)
        batch_size, _, freq_bins, time_frames = x.size()

        x = self.freq_conv(x)  # [B, 16, F, T]

        x = x.permute(0, 3, 1, 2)  # [B, T, 16, F]
        x = x.reshape(batch_size, time_frames, -1)  # [B, T, 16*F]

        lstm_out, _ = self.lstm(x)  # [B, T, hidden_size*2]

        out = self.output_layer(lstm_out)  # [B, T, 16*F]

        out = out.reshape(batch_size, time_frames, 16, freq_bins)  # [B, T, 16, F]
        out = out.permute(0, 2, 3, 1)  # [B, 16, F, T]

        out = self.reconstruct(out)  # [B, 1, F, T]
        enhanced_magnitude = out.squeeze(1)  # [B, F, T]

        return enhanced_magnitude

def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1
    return pesq_score

def pesq_score(utts_r, utts_g, sampling_rate=16000):
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy() if torch.is_tensor(utts_r[i]) else utts_r[i],
                            utts_g[i].squeeze().cpu().numpy() if torch.is_tensor(utts_g[i]) else utts_g[i],
                            sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_scores)
    return pesq_score

def load_agc_model(model_path, device):
    """
    Load a trained standalone AGC model.
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model = AGCModule(input_channels=1, hidden_size=256, num_layers=2, bidirectional=True, freq_bins=201)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    model_info = {
        'n_fft': checkpoint.get('n_fft', 400),
        'stft_hop': checkpoint.get('stft_hop', 100),
        'win_size': checkpoint.get('win_size', 400),
        'test_pesq': checkpoint.get('test_pesq', 'N/A'),
        'test_loss': checkpoint.get('test_loss', 'N/A')
    }
    
    print("Model loaded successfully")
    print(f"Test PESQ: {model_info['test_pesq']}")
    print(f"Test loss: {model_info['test_loss']}")
    
    return model, model_info

def process_audio_file(model, input_file, output_file, n_fft=400, stft_hop=100, win_size=400, 
                      chunk_length=32000, overlap_ratio=0.5, batch_size=64):
    """
    Process one audio file, with chunking for long inputs.
    """
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    original_length = len(audio)
    if len(audio) <= chunk_length:
        enhanced_audio = process_single_chunk(model, audio, n_fft, stft_hop, win_size)
    else:
        enhanced_audio = process_long_audio_chunks(model, audio, chunk_length, overlap_ratio, 
                                                 batch_size, n_fft, stft_hop, win_size)
    sf.write(output_file, enhanced_audio, sr)
    
    return enhanced_audio, sr

def process_single_chunk(model, audio, n_fft=400, stft_hop=100, win_size=400):
    """
    Process a single audio chunk.
    """
    audio_tensor = torch.FloatTensor(audio)
    norm_factor = torch.sqrt(len(audio_tensor) / torch.sum(audio_tensor ** 2.0))
    audio_normalized = audio_tensor * norm_factor
    
    stft = librosa.stft(audio_normalized.numpy(), n_fft=n_fft, hop_length=stft_hop, win_length=win_size)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    magnitude_tensor = torch.FloatTensor(magnitude).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced_magnitude = model(magnitude_tensor)
    enhanced_magnitude_np = enhanced_magnitude[0].cpu().numpy()
    enhanced_stft = enhanced_magnitude_np * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=stft_hop, win_length=win_size, length=len(audio_normalized))
    peak_value = np.max(np.abs(enhanced_audio))
    if peak_value > 0:
        enhanced_audio = enhanced_audio * (0.4 / peak_value)
    
    return enhanced_audio

def split_audio_chunks(audio, chunk_length, overlap_ratio=0.5):
    """
    Split audio into overlapping chunks.
    Args:
        chunk_info: [(start, actual_length), ...]
    """
    chunks = []
    chunk_info = []
    
    overlap_length = int(chunk_length * overlap_ratio)
    step_size = chunk_length - overlap_length
    
    start = 0
    while start < len(audio):
        end = min(start + chunk_length, len(audio))
        chunk = audio[start:end]
        actual_length = len(chunk)
        
        if len(chunk) < chunk_length:
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
        
        chunks.append(chunk)
        chunk_info.append((start, actual_length))
        
        if end >= len(audio):
            break
            
        start += step_size
    
    return chunks, chunk_info

def merge_audio_chunks(enhanced_chunks, chunk_info, original_length, chunk_length, overlap_ratio=0.5):
    """
    Merge processed chunks by averaging overlaps.
    """
    merged = np.zeros(original_length)
    count = np.zeros(original_length)
    
    overlap_length = int(chunk_length * overlap_ratio)
    step_size = chunk_length - overlap_length
    
    for i, (enhanced_chunk, (start, actual_length)) in enumerate(zip(enhanced_chunks, chunk_info)):
        end = start + actual_length
        chunk_data = enhanced_chunk[:actual_length]
        
        merged[start:end] += chunk_data
        count[start:end] += 1
    count[count == 0] = 1
    merged = merged / count
    
    return merged

def process_chunks_batch(model, chunks, batch_size, n_fft=400, stft_hop=100, win_size=400):
    """
    Process audio chunks in batches.
    """
    enhanced_chunks = []
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_enhanced = []
        
        for chunk in batch_chunks:
            chunk_tensor = torch.FloatTensor(chunk)
            norm_factor = torch.sqrt(len(chunk_tensor) / torch.sum(chunk_tensor ** 2.0))
            chunk_normalized = chunk_tensor * norm_factor
            
            stft = librosa.stft(chunk_normalized.numpy(), n_fft=n_fft, hop_length=stft_hop, win_length=win_size)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            magnitude_tensor = torch.FloatTensor(magnitude).unsqueeze(0).to(device)
            with torch.no_grad():
                enhanced_magnitude = model(magnitude_tensor)
            enhanced_magnitude_np = enhanced_magnitude[0].cpu().numpy()
            enhanced_stft = enhanced_magnitude_np * np.exp(1j * phase)
            enhanced_chunk = librosa.istft(enhanced_stft, hop_length=stft_hop, win_length=win_size, length=len(chunk_normalized))
            peak_value = np.max(np.abs(enhanced_chunk))
            if peak_value > 0:
                enhanced_chunk = enhanced_chunk * (0.4 / peak_value)
            
            batch_enhanced.append(enhanced_chunk)
        
        enhanced_chunks.extend(batch_enhanced)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return enhanced_chunks

def process_long_audio_chunks(model, audio, chunk_length, overlap_ratio, batch_size, n_fft, stft_hop, win_size):
    """
    Process long audio with chunking.
    """
    chunks, chunk_info = split_audio_chunks(audio, chunk_length, overlap_ratio)
    enhanced_chunks = process_chunks_batch(model, chunks, batch_size, n_fft, stft_hop, win_size)
    enhanced_audio = merge_audio_chunks(enhanced_chunks, chunk_info, len(audio), chunk_length, overlap_ratio)
    
    return enhanced_audio

def process_single_audio_parallel(args):
    """
    Wrapper for parallel single-file processing.
    """
    model_state_dict, audio_file, output_file, model_info, device_id = args
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id >= 0 else "cpu")
    model = AGCModule(input_channels=1, hidden_size=256, num_layers=2, bidirectional=True, freq_bins=201)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    try:
        enhanced_audio, sr = process_audio_file(
            model, audio_file, output_file,
            n_fft=model_info['n_fft'],
            stft_hop=model_info['stft_hop'], 
            win_size=model_info['win_size']
        )
        return True, os.path.basename(audio_file)
    except Exception as e:
        return False, f"Error processing {os.path.basename(audio_file)}: {str(e)}"

def process_single_audio_joblib(model_state_dict, audio_file, output_file, model_info):
    """
    Single-file joblib worker.
    """
    model = AGCModule(input_channels=1, hidden_size=256, num_layers=2, bidirectional=True, freq_bins=201)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    try:
        enhanced_audio, sr = process_audio_file(
            model, audio_file, output_file,
            n_fft=model_info['n_fft'],
            stft_hop=model_info['stft_hop'], 
            win_size=model_info['win_size']
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, os.path.basename(audio_file)
    except Exception as e:
        return False, f"Error processing {os.path.basename(audio_file)}: {str(e)}"

def batch_process_audio_files_parallel(model, input_dir, output_dir, model_info, num_workers=4):
    """
    Batch process audio files in parallel with joblib.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files. Using {num_workers} workers")
    
    model_state_dict = model.state_dict()
    output_files = [os.path.join(output_dir, os.path.basename(af)) for af in audio_files]
    results = Parallel(n_jobs=num_workers, backend='threading')(
        delayed(process_single_audio_joblib)(model_state_dict, audio_file, output_file, model_info)
        for audio_file, output_file in tqdm(zip(audio_files, output_files), 
                                          total=len(audio_files), desc="Processing audio files")
    )
    success_count = sum(1 for success, _ in results if success)
    error_count = len(results) - success_count
    print(f"Processing complete. Outputs saved to: {output_dir}")
    print(f"Succeeded: {success_count}, Failed: {error_count}")
    for success, message in results:
        if not success:
            print(f"Error: {message}")

def batch_process_audio_files(model, input_dir, output_dir, model_info):
    """
    Batch process audio files serially.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        filename = os.path.basename(audio_file)
        output_file = os.path.join(output_dir, filename)
        enhanced_audio, sr = process_audio_file(
            model, audio_file, output_file, 
            n_fft=model_info['n_fft'],
            stft_hop=model_info['stft_hop'], 
            win_size=model_info['win_size']
        )
    
    print(f"Processing complete. Outputs saved to: {output_dir}")

def main_single_file(input_file, output_file, model_path=None):
    """
    Single-file inference entry point.
    """
    if model_path is None:
        model_path = "/home/jinming/MP-SENet/single_agc_separate_rms/model/agc_model_best.pt"

    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        return

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        print("Train the model first or check the model path")
        return

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model, model_info = load_agc_model(model_path, device)

    print(f"Processing audio file: {input_file}")
    enhanced_audio, sr = process_audio_file(
        model, input_file, output_file,
        n_fft=model_info['n_fft'],
        stft_hop=model_info['stft_hop'],
        win_size=model_info['win_size']
    )

    print(f"Processing complete. Output saved to: {output_file}")
    print(f"Audio length: {len(enhanced_audio)/sr:.2f} seconds")

    return enhanced_audio, sr

def main():
    """
    Batch inference entry point.
    """
    model_path = "/home/users/ntu/ccdsjmzh/scratch/MP-SENet/single_agc_separate_rms/model/agc_model_best.pt"

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        print("Train the model first or check the model path")
        return

    model, model_info = load_agc_model(model_path, device)

    input_dir = "/home/users/ntu/ccdsjmzh/scratch/LibriTTS/test_new_5_30/processed/se_target"
    output_dir = "/home/users/ntu/ccdsjmzh/scratch/LibriTTS/test_new_5_30/processed/se_myagc"

    if torch.cuda.is_available():
        num_workers = 64
    else:
        num_workers = min(8, os.cpu_count())
    
    print(f"Using {num_workers} workers")
    batch_process_audio_files_parallel(model, input_dir, output_dir, model_info, num_workers)

    print("Inference complete")

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("Running batch mode...")
        main()
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        print("Running single-file mode...")
        main_single_file(input_file, output_file)
    elif len(sys.argv) == 4:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        model_path = sys.argv[3]
        print("Running single-file mode with a custom model...")
        main_single_file(input_file, output_file, model_path)
    else:
        print("Usage:")
        print("1. Batch mode: python inference.py")
        print("2. Single-file mode: python inference.py <input_file> <output_file>")
        print("3. Single-file mode with a custom model: python inference.py <input_file> <output_file> <model_path>")
        print("")
        print("Examples:")
        print("python /home/jinming/MP-SENet/single_agc_separate_rms/inference.py")
        print("python /home/jinming/MP-SENet/single_agc_separate_rms/inference.py input.wav output.wav")
        print("python /home/jinming/MP-SENet/single_agc_separate_rms/inference.py input.wav output.wav model.pt")



'''

conda activate diarizen
python /home/users/ntu/ccdsjmzh/scratch/MP-SENet/single_agc_separate_rms/inference.py

'''
