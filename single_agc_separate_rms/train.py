import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
from tqdm import tqdm
import glob
import soundfile as sf
from pesq import pesq
from joblib import Parallel, delayed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PESQ evaluation functions
def cal_pesq(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=15)(delayed(cal_pesq)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)

def pesq_score(utts_r, utts_g, sampling_rate=16000):
    pesq_scores = Parallel(n_jobs=30)(delayed(eval_pesq)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(),
                            sampling_rate)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_scores)
    return pesq_score

def eval_pesq(clean_utt, esti_utt, sr):
    try:
        pesq_score = pesq(sr, clean_utt, esti_utt)
    except:
        pesq_score = -1
    return pesq_score

def asymmetric_magnitude_loss(pred_mag, target_mag, alpha=10.0, silence_threshold=1e-4):
    """
    AGC loss with extra penalty for positive residual energy in silent regions.
    Args:
        pred_mag: Predicted magnitude [B, F, T]
        target_mag: Target magnitude [B, F, T]
        alpha: Penalty factor in silent regions
        silence_threshold: Silence threshold
    Returns:
        Scalar AGC loss
    """
    base_loss = torch.abs(pred_mag - target_mag)
    silence_mask = target_mag <= silence_threshold
    noise_in_silence = silence_mask & (pred_mag > silence_threshold)
    loss_weights = torch.ones_like(base_loss)
    loss_weights = torch.where(noise_in_silence, alpha, loss_weights)
    weighted_loss = base_loss * loss_weights
    return torch.mean(weighted_loss)

class SpectralAGCDatasetSeparateRMS(Dataset):
    def __init__(self, lower_dir, origin_dir, segment_length=32000, hop_length=16000,
                 n_fft=400, stft_hop=100, win_size=400):
        """
        Frequency-domain AGC dataset.

        Args:
            lower_dir: Low-gain audio directory
            origin_dir: Target audio directory
        """
        self.lower_dir = lower_dir
        self.origin_dir = origin_dir
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.stft_hop = stft_hop
        self.win_size = win_size

        self.lower_files = sorted(glob.glob(os.path.join(lower_dir, "*.wav")))
        self.origin_files = sorted(glob.glob(os.path.join(origin_dir, "*.wav")))

        assert len(self.lower_files) == len(self.origin_files), "File count mismatch"

        self.segments = []
        for idx, (lower_file, origin_file) in enumerate(zip(self.lower_files, self.origin_files)):
            lower_basename = os.path.basename(lower_file)
            origin_basename = os.path.basename(origin_file)
            assert lower_basename == origin_basename, f"Filename mismatch: {lower_basename} vs {origin_basename}"

            audio_info = sf.info(lower_file)
            audio_length = int(audio_info.frames)
            num_segments = max(1, (audio_length - segment_length) // hop_length + 1)
            for i in range(num_segments):
                start_sample = i * hop_length
                self.segments.append((idx, start_sample))
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        file_idx, start_sample = self.segments[idx]

        lower_file = self.lower_files[file_idx]
        origin_file = self.origin_files[file_idx]

        lower_audio, sr = sf.read(lower_file, start=start_sample, stop=start_sample + self.segment_length, dtype='float32')
        origin_audio, _ = sf.read(origin_file, start=start_sample, stop=start_sample + self.segment_length, dtype='float32')

        if len(lower_audio) < self.segment_length:
            lower_audio = np.pad(lower_audio, (0, self.segment_length - len(lower_audio)))
            origin_audio = np.pad(origin_audio, (0, self.segment_length - len(origin_audio)))

        lower_audio_tensor = torch.FloatTensor(lower_audio)
        origin_audio_tensor = torch.FloatTensor(origin_audio)

        norm_factor = torch.sqrt(len(lower_audio_tensor) / torch.sum(lower_audio_tensor ** 2.0))
        lower_audio_normalized = lower_audio_tensor * norm_factor
        origin_audio_raw = origin_audio_tensor

        lower_stft = librosa.stft(lower_audio_normalized.numpy(), n_fft=self.n_fft, hop_length=self.stft_hop, win_length=self.win_size)
        origin_stft = librosa.stft(origin_audio_raw.numpy(), n_fft=self.n_fft, hop_length=self.stft_hop, win_length=self.win_size)

        lower_mag = np.abs(lower_stft)
        origin_mag = np.abs(origin_stft)

        lower_mag_tensor = torch.FloatTensor(lower_mag)
        origin_mag_tensor = torch.FloatTensor(origin_mag)

        lower_phase_tensor = torch.FloatTensor(np.angle(lower_stft))

        return (lower_mag_tensor, origin_mag_tensor, lower_phase_tensor, 
                lower_audio_normalized, origin_audio_raw, 
                norm_factor, origin_audio_tensor,
                lower_audio_tensor, origin_audio_tensor)

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
        Returns:
            enhanced_magnitude: [B, F, T]
        """
        x = magnitude.unsqueeze(1)
        batch_size, _, freq_bins, time_frames = x.size()
        x = self.freq_conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time_frames, -1)
        lstm_out, _ = self.lstm(x)
        out = self.output_layer(lstm_out)
        out = out.reshape(batch_size, time_frames, 16, freq_bins)
        out = out.permute(0, 2, 3, 1)
        out = self.reconstruct(out)
        enhanced_magnitude = out.squeeze(1)

        return enhanced_magnitude

def train_agc_model(model, train_loader, val_loader, num_epochs=400, lr=0.0005, patience=10, output_dir=".", n_fft=400, stft_hop=100, win_size=400, alpha=10.0):
    """
    Train the standalone AGC model.

    Args:
        alpha: Silent-region penalty factor
    """
    def criterion(pred, target):
        return asymmetric_magnitude_loss(pred, target, alpha=alpha)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=[0.8, 0.99])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pesq': []
    }

    best_val_pesq = -float('inf')
    best_model_state = None
    early_stop_counter = 0

    val_results_file = os.path.join(output_dir, "validation_results.txt")
    with open(val_results_file, "w", encoding="utf-8") as f:
        f.write("AGC Model Training Validation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Training Parameters:\n")
        f.write(f"- Learning Rate: {lr}\n")
        f.write(f"- Epochs: {num_epochs}\n")
        f.write(f"- Patience: {patience}\n")
        f.write(f"- Alpha (silence noise penalty): {alpha}\n")
        f.write(f"- STFT Parameters: n_fft={n_fft}, hop={stft_hop}, win_size={win_size}\n")
        f.write("="*50 + "\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            (lower_mag, origin_mag, lower_phase,
             lower_audio_norm, origin_audio_raw,
             norm_factor, _,
             lower_audio_orig, origin_audio_orig) = batch_data

            lower_mag = lower_mag.to(device)
            origin_mag = origin_mag.to(device)

            optimizer.zero_grad()
            outputs = model(lower_mag)
            batch_size, freq_bins, time_frames = origin_mag.shape
            origin_rms = torch.sqrt(torch.mean(origin_mag.view(batch_size, -1) ** 2, dim=1))
            origin_norm_factor = 1.0 / (origin_rms + 1e-8)
            origin_norm_factor_expanded = origin_norm_factor.view(-1, 1, 1)
            origin_mag_normalized = origin_mag * origin_norm_factor_expanded
            loss = criterion(outputs, origin_mag_normalized)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                (lower_mag, origin_mag, lower_phase,
                 lower_audio_norm, origin_audio_raw,
                 norm_factor, _,
                 lower_audio_orig, origin_audio_orig) = batch_data

                lower_mag = lower_mag.to(device)
                origin_mag = origin_mag.to(device)

                outputs = model(lower_mag)
                batch_size, freq_bins, time_frames = origin_mag.shape
                origin_rms = torch.sqrt(torch.mean(origin_mag.view(batch_size, -1) ** 2, dim=1))
                origin_norm_factor = 1.0 / (origin_rms + 1e-8)
                origin_norm_factor_expanded = origin_norm_factor.view(-1, 1, 1)
                origin_mag_normalized = origin_mag * origin_norm_factor_expanded
                
                loss = criterion(outputs, origin_mag_normalized)
                
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)

        if hasattr(val_loader.dataset, 'dataset'):
            original_dataset = val_loader.dataset.dataset.dataset if hasattr(val_loader.dataset.dataset, 'dataset') else val_loader.dataset.dataset
        else:
            original_dataset = val_loader.dataset
            
        val_pesq = evaluate_whole_audio_pesq(model, original_dataset, num_files=100, 
                                           n_fft=n_fft, stft_hop=stft_hop, win_size=win_size)
        history['val_pesq'].append(val_pesq)

        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Val PESQ: {val_pesq:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        val_results_file = os.path.join(output_dir, "validation_results.txt")
        with open(val_results_file, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, "
                   f"Train Loss: {avg_train_loss:.6f}, "
                   f"Val Loss: {avg_val_loss:.6f}, "
                   f"Val PESQ: {val_pesq:.4f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")

        if val_pesq > best_val_pesq:
            best_val_pesq = val_pesq
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0

            best_model_path = os.path.join(output_dir, "agc_model_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_pesq': val_pesq
            }, best_model_path)
            print(f"Saved best model to: {best_model_path} (PESQ: {val_pesq:.4f})")
        else:
            early_stop_counter += 1
            print(f"Validation PESQ did not improve. Early-stop counter: {early_stop_counter}/{patience}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"agc_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_pesq': val_pesq
            }, checkpoint_path)

        if early_stop_counter >= patience:
            print(f"Early stopping: validation PESQ did not improve for {patience} epochs")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best validation PESQ model (Val PESQ: {best_val_pesq:.4f})")

    val_results_file = os.path.join(output_dir, "validation_results.txt")
    with open(val_results_file, "a", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("Training Summary:\n")
        f.write(f"- Total Epochs: {epoch+1}\n")
        f.write(f"- Best Validation PESQ: {best_val_pesq:.4f}\n")
        f.write(f"- Final Train Loss: {avg_train_loss:.6f}\n")
        f.write(f"- Final Val Loss: {avg_val_loss:.6f}\n")
        if early_stop_counter >= patience:
            f.write(f"- Early Stopped: Yes (patience reached)\n")
        else:
            f.write(f"- Early Stopped: No\n")
        f.write("="*50 + "\n")

    return history, best_val_pesq

def evaluate_whole_audio_pesq(model, dataset, num_files=10, n_fft=400, stft_hop=100, win_size=400):
    """
    Evaluate PESQ on whole audio files.
    
    Args:
        num_files: Number of files to evaluate
    """
    model.eval()
    pesq_scores = []
    
    file_pairs = list(zip(dataset.lower_files, dataset.origin_files))[:num_files]
    
    with torch.no_grad():
        for lower_file, origin_file in tqdm(file_pairs, desc="Evaluating whole audio PESQ"):
            try:
                lower_audio, sr = sf.read(lower_file, dtype='float32')
                origin_audio, _ = sf.read(origin_file, dtype='float32')
                min_length = min(len(lower_audio), len(origin_audio))
                lower_audio = lower_audio[:min_length]
                origin_audio = origin_audio[:min_length]
                
                lower_audio_tensor = torch.FloatTensor(lower_audio)
                norm_factor = torch.sqrt(len(lower_audio_tensor) / torch.sum(lower_audio_tensor ** 2.0))
                lower_audio_normalized = lower_audio_tensor * norm_factor
                
                lower_stft = librosa.stft(lower_audio_normalized.numpy(), 
                                        n_fft=n_fft, hop_length=stft_hop, win_length=win_size)
                lower_mag = np.abs(lower_stft)
                lower_phase = np.angle(lower_stft)
                
                lower_mag_tensor = torch.FloatTensor(lower_mag).unsqueeze(0).to(device)
                enhanced_mag = model(lower_mag_tensor)
                enhanced_mag_np = enhanced_mag[0].cpu().numpy()
                
                enhanced_stft = enhanced_mag_np * np.exp(1j * lower_phase)
                enhanced_audio = librosa.istft(enhanced_stft, hop_length=stft_hop, 
                                             win_length=win_size, length=len(lower_audio_normalized))
                
                peak_value = np.max(np.abs(enhanced_audio))
                if peak_value > 0:
                    enhanced_audio = enhanced_audio * (0.4 / peak_value)
                
                origin_peak = np.max(np.abs(origin_audio))
                if origin_peak > 0:
                    origin_audio_norm = origin_audio * (0.4 / origin_peak)
                else:
                    origin_audio_norm = origin_audio
                
                pesq_val = eval_pesq(origin_audio_norm, enhanced_audio, sr)
                if pesq_val != -1:
                    pesq_scores.append(pesq_val)
                    
            except Exception as e:
                print(f"Error processing {lower_file}: {e}")
                continue
    
    if pesq_scores:
        avg_pesq = np.mean(pesq_scores)
        print(f"Whole-audio PESQ complete. Average PESQ: {avg_pesq:.4f} ({len(pesq_scores)} files)")
        return avg_pesq
    else:
        print("No valid PESQ scores")
        return 0.0

def evaluate_agc_model(model, test_loader, n_fft=400, stft_hop=100, win_size=400, output_dir=".", alpha=10.0):
    """
    Evaluate the standalone AGC model.
    """
    model.eval()
    def criterion(pred, target):
        return asymmetric_magnitude_loss(pred, target, alpha=alpha)
    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            (lower_mag, origin_mag, lower_phase,
             lower_audio_norm, origin_audio_raw,
             norm_factor, _,
             lower_audio_orig, origin_audio_orig) = batch_data

            lower_mag = lower_mag.to(device)
            origin_mag = origin_mag.to(device)
            lower_phase = lower_phase.to(device)

            outputs = model(lower_mag)
            batch_size, freq_bins, time_frames = origin_mag.shape
            origin_rms = torch.sqrt(torch.mean(origin_mag.view(batch_size, -1) ** 2, dim=1))
            origin_norm_factor = 1.0 / (origin_rms + 1e-8)
            origin_norm_factor_expanded = origin_norm_factor.view(-1, 1, 1)
            origin_mag_normalized = origin_mag * origin_norm_factor_expanded
            
            loss = criterion(outputs, origin_mag_normalized)
            
            test_loss += loss.item()
            num_batches += 1

            if i < 5:
                enhanced_stft = outputs[0].cpu().numpy() * np.exp(1j * lower_phase[0].cpu().numpy())
                enhanced_audio = librosa.istft(enhanced_stft, hop_length=stft_hop, win_length=win_size, length=len(lower_audio_norm[0]))
                peak_value = np.max(np.abs(enhanced_audio))
                if peak_value > 0:
                    enhanced_audio = enhanced_audio * (0.4 / peak_value)
                
                sf.write(os.path.join(output_dir, f'test_input_{i}.wav'), lower_audio_orig[0].numpy(), 16000)
                sf.write(os.path.join(output_dir, f'test_output_{i}.wav'), enhanced_audio, 16000)
                sf.write(os.path.join(output_dir, f'test_target_{i}.wav'), origin_audio_orig[0].numpy(), 16000)

    avg_test_loss = test_loss / num_batches

    if hasattr(test_loader.dataset, 'dataset'):
        original_dataset = test_loader.dataset.dataset.dataset if hasattr(test_loader.dataset.dataset, 'dataset') else test_loader.dataset.dataset
    else:
        original_dataset = test_loader.dataset
        
    test_pesq = evaluate_whole_audio_pesq(model, original_dataset, num_files=10, 
                                        n_fft=n_fft, stft_hop=stft_hop, win_size=win_size)
    
    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test PESQ: {test_pesq:.4f}")
    
    test_results_file = os.path.join(output_dir, "test_results.txt")
    with open(test_results_file, "w", encoding="utf-8") as f:
        f.write("AGC Model Test Results\n")
        f.write("="*30 + "\n")
        f.write(f"Test Loss: {avg_test_loss:.6f}\n")
        f.write(f"Test PESQ: {test_pesq:.4f}\n")
        f.write(f"Evaluation method: Whole audio files\n")
        f.write("="*30 + "\n")

    return avg_test_loss, test_pesq

def process_audio_file_agc(model, input_file, output_file, n_fft=400, stft_hop=100, win_size=400):
    """
    Process a full audio file with the standalone AGC model.
    """
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    audio_tensor = torch.FloatTensor(audio)
    norm_factor = torch.sqrt(len(audio_tensor) / torch.sum(audio_tensor ** 2.0))
    audio_normalized = audio_tensor * norm_factor

    stft = librosa.stft(audio_normalized.numpy(), n_fft=n_fft, hop_length=stft_hop, win_length=win_size)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    magnitude_tensor = torch.FloatTensor(magnitude).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        enhanced_magnitude = model(magnitude_tensor)
    enhanced_magnitude_np = enhanced_magnitude[0].cpu().numpy()
    enhanced_stft = enhanced_magnitude_np * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=stft_hop, win_length=win_size, length=len(audio_normalized))
    peak_value = np.max(np.abs(enhanced_audio))
    if peak_value > 0:
        enhanced_audio = enhanced_audio * (0.4 / peak_value)

    sf.write(output_file, enhanced_audio, sr)

    return enhanced_audio, sr

def main(output_dir="./agc_results"):
    """
    Train and evaluate the standalone AGC model.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving standalone AGC artifacts to: {output_dir}")

    data_dir = "/home/users/ntu/ccdsjmzh/scratch/LibriTTS/train_new_5_30"
    lower_dir = os.path.join(data_dir, "lower")
    origin_dir = os.path.join(data_dir, "origin")

    if not os.path.exists(lower_dir) or not os.path.exists(origin_dir):
        raise FileNotFoundError(f"Directory does not exist: {lower_dir} or {origin_dir}")

    n_fft = 400
    stft_hop = 100
    win_size = 400

    segment_length = 32000
    dataset = SpectralAGCDatasetSeparateRMS(lower_dir, origin_dir, segment_length=segment_length,
                                           n_fft=n_fft, stft_hop=stft_hop, win_size=win_size)

    total_size = len(dataset)
    subset_size = int(1 * total_size)
    print(f"Original dataset size: {total_size}")
    print(f"Subset size: {subset_size}")

    indices = torch.randperm(total_size)[:subset_size]
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    subset_total_size = len(subset_dataset)
    train_size = int(0.9 * subset_total_size)
    val_size = int(0.05 * subset_total_size)
    test_size = subset_total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        subset_dataset, [train_size, val_size, test_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16)

    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    model = AGCModule(input_channels=1, hidden_size=256, num_layers=2, bidirectional=True, freq_bins=201).to(device)

    num_epochs = 400
    patience = 10
    history, best_val_pesq = train_agc_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        lr=0.001,
        patience=patience,
        output_dir=output_dir,
        n_fft=n_fft,
        stft_hop=stft_hop,
        win_size=win_size
    )

    test_loss, test_pesq = evaluate_agc_model(model, test_loader, n_fft=n_fft, stft_hop=stft_hop, win_size=win_size, output_dir=output_dir, alpha=10.0)

    torch.save({
        'model_state_dict': model.state_dict(),
        'test_loss': test_loss,
        'test_pesq': test_pesq,
        'n_fft': n_fft,
        'stft_hop': stft_hop,
        'win_size': win_size,
        'history': history
    }, os.path.join(output_dir, "agc_model_final.pt"))

    print("Standalone AGC training and evaluation complete")
    print(f"Best validation PESQ: {best_val_pesq:.4f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test PESQ: {test_pesq:.4f}")
    print(f"All model files saved to: {output_dir}")

    summary_file = os.path.join(output_dir, "training_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("AGC Model Training & Testing Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset Information:\n")
        f.write(f"- Total original dataset size: {total_size}\n")
        f.write(f"- Used subset size (10%): {subset_size}\n")
        f.write(f"- Training set: {len(train_dataset)}\n")
        f.write(f"- Validation set: {len(val_dataset)}\n")
        f.write(f"- Test set: {len(test_dataset)}\n")
        f.write(f"\nModel Configuration:\n")
        f.write(f"- Architecture: AGC Module\n")
        f.write(f"- Hidden size: 256\n")
        f.write(f"- Num layers: 2\n")
        f.write(f"- Bidirectional: True\n")
        f.write(f"- Frequency bins: 201\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"- Epochs: {num_epochs}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Learning rate: 0.0005\n")
        f.write(f"- Patience: {patience}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"- Best Validation PESQ: {best_val_pesq:.4f}\n")
        f.write(f"- Test Loss: {test_loss:.6f}\n")
        f.write(f"- Test PESQ: {test_pesq:.4f}\n")
        f.write(f"\nOutput Files:\n")
        f.write(f"- Model files: {output_dir}/agc_model_*.pt\n")
        f.write(f"- Validation log: {output_dir}/validation_results.txt\n")
        f.write(f"- Test results: {output_dir}/test_results.txt\n")
        f.write(f"- Sample outputs: {output_dir}/test_*.wav\n")
        f.write("="*50 + "\n")

    return model

if __name__ == "__main__":
    main(output_dir="/home/users/ntu/ccdsjmzh/scratch/MP-SENet/single_agc_separate_rms/model2")

'''
Usage:
conda activate mpsenet
python /home/users/ntu/ccdsjmzh/scratch/MP-SENet/single_agc_separate_rms/train.py
'''

