"""MP-SENet: Magnitude and Phase Speech Enhancement Network."""

import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerBlock
from utils import LearnableSigmoid2d
from pesq import pesq
from joblib import Parallel, delayed


class SPConvTranspose2d(nn.Module):
    """Sub-pixel convolutional transpose for upsampling."""
    
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    """Dense block with dilated convolutions."""
    
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(h.dense_channel*(i+1), h.dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """Dense encoder for feature extraction."""
    
    def __init__(self, h, in_channel):
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))
        self.dense_block = DenseBlock(h, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)
        x = self.dense_block(x)
        x = self.dense_conv_2(x)
        return x


class MaskDecoder(nn.Module):
    """Decoder for magnitude mask estimation."""
    
    def __init__(self, h, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft//2+1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    """Decoder for phase estimation."""
    
    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        
    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        return x


class TSTransformerBlock(nn.Module):
    """Time-frequency transformer block."""
    
    def __init__(self, h):
        super(TSTransformerBlock, self).__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    """Magnitude and Phase Speech Enhancement Network."""
    
    def __init__(self, h, num_tsblocks=4):
        super(MPNet, self).__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)
        self.TSTransformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))
        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)
        x = self.dense_encoder(x)
        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)
        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack((denoised_amp*torch.cos(denoised_pha),
                                    denoised_amp*torch.sin(denoised_pha)), dim=-1)
        return denoised_amp, denoised_pha, denoised_com


def phase_losses(phase_r, phase_g):
    """Calculate phase losses: instantaneous phase, group delay, instantaneous angular frequency."""
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip_loss, gd_loss, iaf_loss


def anti_wrapping_function(x):
    """Anti-wrapping function for phase."""
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


