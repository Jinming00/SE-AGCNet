"""Utility functions for SE-AGCNet."""

import os
import glob
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")

def get_padding(kernel_size, dilation=1):
    """Calculate padding for 1D convolution."""
    return int((kernel_size*dilation - dilation)/2)


class LearnableSigmoid1d(nn.Module):
    """Learnable sigmoid activation for 1D tensors."""
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    """Learnable sigmoid activation for 2D tensors."""
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


def load_checkpoint(filepath, device):
    """Load checkpoint from file."""
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """Save checkpoint to file."""
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    """Scan and return the latest checkpoint."""
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
