"""Automatic Gain Control (AGC) module using BiLSTM."""

import torch
import torch.nn as nn


class AGCModule(nn.Module):
    """
    AGC module for dynamic range compression using BiLSTM.
    Performs RMS normalization and level control on magnitude spectrograms.
    """
    
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2, 
                 bidirectional=True, freq_bins=257):
        super(AGCModule, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.freq_bins = freq_bins
        
        # Convolutional encoder
        self.freq_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=16 * freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Output layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * freq_bins)
        )
        
        # Convolutional decoder to reconstruct magnitude
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
        x = x.permute(0, 3, 1, 2).reshape(batch_size, time_frames, -1)
        lstm_out, _ = self.lstm(x)
        out = self.output_layer(lstm_out)
        out = out.reshape(batch_size, time_frames, 16, freq_bins).permute(0, 2, 3, 1)
        out = self.reconstruct(out)
        enhanced_magnitude = out.squeeze(1)
        
        return enhanced_magnitude


class MPSENetAGC(nn.Module):
    """Combined MP-SENet + AGC model for speech enhancement."""
    
    def __init__(self, h, num_tsblocks=4):
        super(MPSENetAGC, self).__init__()
        from .model import MPNet
        
        self.h = h
        self.mpnet = MPNet(h, num_tsblocks)
        self.agc = AGCModule(freq_bins=h.n_fft // 2 + 1)
    
    def forward(self, noisy_amp, noisy_pha, norm_factor=None):
        """
        Args:
            noisy_amp: [B, F, T] Noisy magnitude
            noisy_pha: [B, F, T] Noisy phase  
            norm_factor: [B] RMS normalization factor
        Returns:
            Tuple of (agc_amp, mpnet_pha, agc_com, mpnet_amp, mpnet_com, agc_norm)
        """
        mpnet_amp, mpnet_pha, mpnet_com = self.mpnet(noisy_amp, noisy_pha)
        
        if norm_factor is not None:
            mpnet_amp_denormalized = mpnet_amp / norm_factor.view(-1, 1, 1)
        else:
            mpnet_amp_denormalized = mpnet_amp
        
        batch_size = mpnet_amp_denormalized.size(0)
        agc_input_rms = torch.sqrt(
            torch.mean(mpnet_amp_denormalized.view(batch_size, -1) ** 2, dim=1))
        agc_norm_factor = 1.0 / (agc_input_rms + 1e-8)
        agc_input_norm = mpnet_amp_denormalized * agc_norm_factor.view(-1, 1, 1)
        
        agc_amp_normalized = self.agc(agc_input_norm)
        
        agc_com_normalized = torch.stack(
            (agc_amp_normalized * torch.cos(mpnet_pha),
             agc_amp_normalized * torch.sin(mpnet_pha)), dim=-1)
        
        return (agc_amp_normalized, mpnet_pha, agc_com_normalized, 
                mpnet_amp, mpnet_com, agc_norm_factor)

