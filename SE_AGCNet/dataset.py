"""Dataset utilities for SE-AGCNet."""

import os
import random
import torch
import torch.utils.data
import librosa


def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """Compute magnitude, phase, and complex spectrogram via STFT."""
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                           window=hann_window, center=center, pad_mode='reflect', 
                           normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-10, stft_spec[:, :, :, 0] + 1e-5)
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """Reconstruct waveform from magnitude and phase via iSTFT."""
    mag = torch.pow(mag, 1/compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=-1)
    hann_window = torch.hann_window(win_size).to(com.device)
    spec = torch.view_as_complex(com)
    wav = torch.istft(spec, n_fft, hop_length=hop_size, win_length=win_size, 
                     window=hann_window, center=center)
    return wav


def get_dataset_filelist(a):
    """Get list of training file indexes from directories."""
    clean_dirs = [d.strip() for d in a.input_train_clean_dir.split(',')]
    noisy_dirs = [d.strip() for d in a.input_train_noisy_dir.split(',')]
    origin_dirs = [d.strip() for d in a.input_train_origin_dir.split(',')]

    if not (len(clean_dirs) == len(noisy_dirs) == len(origin_dirs)):
        raise ValueError(
            f"Directory count mismatch: clean={len(clean_dirs)}, "
            f"noisy={len(noisy_dirs)}, origin={len(origin_dirs)}"
        )

    print(f"Processing {len(clean_dirs)} training directory sets:")
    for i, (clean_dir, noisy_dir, origin_dir) in enumerate(
        zip(clean_dirs, noisy_dirs, origin_dirs)):
        print(f"  Set {i+1}: {clean_dir}")
        print(f"         {noisy_dir}")
        print(f"         {origin_dir}")

    all_training_indexes = []
    for clean_dir, noisy_dir, origin_dir in zip(clean_dirs, noisy_dirs, origin_dirs):
        train_clean_files = [f[:-4] for f in os.listdir(clean_dir) if f.endswith('.wav')]
        train_noisy_files = [f[:-4] for f in os.listdir(noisy_dir) if f.endswith('.wav')]
        train_origin_files = [f[:-4] for f in os.listdir(origin_dir) if f.endswith('.wav')]

        common_files = list(set(train_clean_files) & set(train_noisy_files) & set(train_origin_files))
        print(f"  Found {len(common_files)} common files")
        all_training_indexes.extend(common_files)

    all_training_indexes = list(set(all_training_indexes))
    random.seed(1234)
    random.shuffle(all_training_indexes)

    print(f"Total training files: {len(all_training_indexes)}")
    return all_training_indexes


class DatasetWithOrigin(torch.utils.data.Dataset):
    """Dataset with clean, noisy, and origin audio."""
    
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, origin_wavs_dir,
                 segment_size, sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)

        self.clean_wavs_dirs = [d.strip() for d in clean_wavs_dir.split(',')]
        self.noisy_wavs_dirs = [d.strip() for d in noisy_wavs_dir.split(',')]
        self.origin_wavs_dirs = [d.strip() for d in origin_wavs_dir.split(',')]

        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.cached_origin_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def _find_file_in_dirs(self, filename, dirs):
        """Find file in multiple directories."""
        for dir_path in dirs:
            file_path = os.path.join(dir_path, filename + '.wav')
            if os.path.exists(file_path):
                return file_path
        raise FileNotFoundError(f"{filename}.wav not found in: {dirs}")

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_file_path = self._find_file_in_dirs(filename, self.clean_wavs_dirs)
            noisy_file_path = self._find_file_in_dirs(filename, self.noisy_wavs_dirs)
            origin_file_path = self._find_file_in_dirs(filename, self.origin_wavs_dirs)

            clean_audio, _ = librosa.load(clean_file_path, sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(noisy_file_path, sr=self.sampling_rate)
            origin_audio, _ = librosa.load(origin_file_path, sr=self.sampling_rate)

            length = min(len(clean_audio), len(noisy_audio), len(origin_audio))
            clean_audio = clean_audio[:length]
            noisy_audio = noisy_audio[:length]
            origin_audio = origin_audio[:length]

            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self.cached_origin_wav = origin_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            origin_audio = self.cached_origin_wav
            self._cache_ref_count -= 1

        if self.split:
            if len(clean_audio) > self.segment_size:
                max_audio_start = len(clean_audio) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[audio_start:audio_start+self.segment_size]
                noisy_audio = noisy_audio[audio_start:audio_start+self.segment_size]
                origin_audio = origin_audio[audio_start:audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(torch.FloatTensor(clean_audio), (0, self.segment_size - len(clean_audio)), 'constant').numpy()
                noisy_audio = torch.nn.functional.pad(torch.FloatTensor(noisy_audio), (0, self.segment_size - len(noisy_audio)), 'constant').numpy()
                origin_audio = torch.nn.functional.pad(torch.FloatTensor(origin_audio), (0, self.segment_size - len(origin_audio)), 'constant').numpy()

        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)
        origin_audio = torch.FloatTensor(origin_audio)

        # Normalize clean and noisy audio
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)
        origin_audio = origin_audio.unsqueeze(0)

        return clean_audio.squeeze(), noisy_audio.squeeze(), origin_audio.squeeze(), norm_factor

    def __len__(self):
        return len(self.audio_indexes)


class Dataset(torch.utils.data.Dataset):
    """Dataset with clean and noisy audio."""
    
    def __init__(self, training_indexes, clean_wavs_dir, noisy_wavs_dir, segment_size, 
                 sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_indexes = training_indexes
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_indexes)
        self.clean_wavs_dir = clean_wavs_dir
        self.noisy_wavs_dir = noisy_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_indexes[index]
        if self._cache_ref_count == 0:
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            noisy_audio, _ = librosa.load(os.path.join(self.noisy_wavs_dir, filename + '.wav'), sr=self.sampling_rate)
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        if self.split:
            if len(clean_audio) > self.segment_size:
                max_audio_start = len(clean_audio) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[audio_start:audio_start+self.segment_size]
                noisy_audio = noisy_audio[audio_start:audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(torch.FloatTensor(clean_audio), (0, self.segment_size - len(clean_audio)), 'constant').numpy()
                noisy_audio = torch.nn.functional.pad(torch.FloatTensor(noisy_audio), (0, self.segment_size - len(noisy_audio)), 'constant').numpy()

        clean_audio = torch.FloatTensor(clean_audio)
        noisy_audio = torch.FloatTensor(noisy_audio)

        return clean_audio, noisy_audio

    def __len__(self):
        return len(self.audio_indexes)
