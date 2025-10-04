"""
Audio Augmentation Module

Augmentation modes for AGC simulation:
- sudden_spikes: Random short volume spikes (2-5x)
- gradual_increase: Progressive volume increase
- gradual_decrease: Progressive volume decrease
- volume_fluctuation: Wave-like volume variations

Each mode: 15% probability (60% total augmentation rate).
"""

import numpy as np
import random


class AudioAugmentation:
    """Audio augmentation class for simulating various volume change scenarios."""
    
    def __init__(self, random_seed=42):
        """
        Initialize audio augmenter.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.modes = [
            'sudden_spikes',
            'gradual_increase',
            'gradual_decrease',
            'volume_fluctuation'
        ]
    
    def apply_augmentation(self, audio, sample_rate, mode=None, probability=0.15):
        """
        Apply augmentation to audio with precise probability control.
        
        Args:
            audio: Audio data (numpy array)
            sample_rate: Sample rate
            mode: Specific mode to apply (None for random selection)
            probability: Application probability for each mode (default 15%)
        
        Returns:
            tuple: (processed_audio, applied_mode)
        """
        if mode is not None:
            processed_audio = self._apply_mode(audio.copy(), mode, sample_rate)
            return processed_audio, mode
        
        r = random.random()
        
        # Probability distribution:
        # 40% no augmentation (0.0-0.4)
        # 15% sudden_spikes (0.4-0.55)
        # 15% gradual_increase (0.55-0.7)
        # 15% gradual_decrease (0.7-0.85)
        # 15% volume_fluctuation (0.85-1.0)
        
        if r < 0.4:
            return audio.copy(), 'none'
        elif r < 0.55:
            return self._apply_mode(audio.copy(), 'sudden_spikes', sample_rate), 'sudden_spikes'
        elif r < 0.7:
            return self._apply_mode(audio.copy(), 'gradual_increase', sample_rate), 'gradual_increase'
        elif r < 0.85:
            return self._apply_mode(audio.copy(), 'gradual_decrease', sample_rate), 'gradual_decrease'
        else:
            return self._apply_mode(audio.copy(), 'volume_fluctuation', sample_rate), 'volume_fluctuation'
    
    def _apply_mode(self, audio, mode, sample_rate):
        """Apply specified augmentation mode."""
        if mode == 'sudden_spikes':
            return self._apply_sudden_spikes(audio, sample_rate)
        elif mode == 'gradual_increase':
            return self._apply_gradual_increase(audio, sample_rate)
        elif mode == 'gradual_decrease':
            return self._apply_gradual_decrease(audio, sample_rate)
        elif mode == 'volume_fluctuation':
            return self._apply_volume_fluctuation(audio, sample_rate)
        return audio
    
    def get_mode_description(self, mode):
        """Get augmentation mode description."""
        descriptions = {
            'none': 'No augmentation - original audio preserved',
            'sudden_spikes': 'Sudden volume spikes - random short-duration volume peaks',
            'gradual_increase': 'Gradual increase - progressive volume increase from low to high',
            'gradual_decrease': 'Gradual decrease - progressive volume decrease from high to low',
            'volume_fluctuation': 'Volume fluctuation - wave-like volume variations'
        }
        return descriptions.get(mode, 'Unknown augmentation mode')
    
    def _apply_sudden_spikes(self, audio, sample_rate, spike_intensity=None, 
                            spike_duration=None, num_spikes=None):
        """
        Apply sudden volume spikes.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            spike_intensity: Spike intensity multiplier (default 2-5x random)
            spike_duration: Spike duration (default 0.1-0.3s random)
            num_spikes: Number of spikes (default 1-3 random)
        """
        if spike_intensity is None:
            spike_intensity = random.uniform(2.0, 5.0)
        if spike_duration is None:
            spike_duration = random.uniform(0.1, 0.3)
        if num_spikes is None:
            num_spikes = random.randint(1, 3)
        
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        
        if len(audio) < frame_length:
            mid_pos = len(audio) // 2
            spike_samples = int(spike_duration * sample_rate)
            start_idx = max(0, mid_pos - spike_samples // 2)
            end_idx = min(len(audio), start_idx + spike_samples)
            audio[start_idx:end_idx] *= spike_intensity
            return audio
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame_energy = np.sum(audio[i:i+frame_length]**2)
            energy.append(frame_energy)
        
        energy = np.array(energy)
        energy_times = np.array([i/sample_rate for i in range(0, len(audio)-frame_length, hop_length)])
        
        if len(energy) > 0:
            energy_threshold = np.percentile(energy, 60)
            active_regions = energy > energy_threshold
            active_times = energy_times[active_regions]
            
            if len(active_times) > 0:
                spike_positions = []
                for _ in range(min(num_spikes, len(active_times))):
                    if len(active_times) > 0:
                        pos = random.choice(active_times)
                        spike_positions.append(pos)
                        active_times = active_times[np.abs(active_times - pos) > spike_duration]
            else:
                duration = len(audio) / sample_rate
                spike_positions = [random.uniform(0, duration - spike_duration) for _ in range(num_spikes)]
        else:
            duration = len(audio) / sample_rate
            spike_positions = [duration * 0.5]
        
        for pos in spike_positions:
            start_idx = int(pos * sample_rate)
            end_idx = int((pos + spike_duration) * sample_rate)
            end_idx = min(end_idx, len(audio))
            
            if start_idx < len(audio):
                audio[start_idx:end_idx] *= spike_intensity
        
        return audio
    
    def _apply_gradual_increase(self, audio, sample_rate, start_multiplier=None, end_multiplier=None):
        """
        Apply gradual volume increase.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            start_multiplier: Starting volume multiplier (default 0.8-1.2x random)
            end_multiplier: Ending volume multiplier (default 2.5-3.5x random)
        """
        if start_multiplier is None:
            start_multiplier = random.uniform(0.8, 1.2)
        if end_multiplier is None:
            end_multiplier = random.uniform(2.5, 3.5)
        
        length = len(audio)
        x = np.linspace(-3, 3, length)
        sigmoid_curve = 1 / (1 + np.exp(-x))
        volume_curve = start_multiplier + (end_multiplier - start_multiplier) * sigmoid_curve
        
        audio *= volume_curve
        return audio
    
    def _apply_gradual_decrease(self, audio, sample_rate, start_multiplier=None, end_multiplier=None):
        """
        Apply gradual volume decrease.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            start_multiplier: Starting volume multiplier (default 1.5-3.0x random)
            end_multiplier: Ending volume multiplier (default 0.2-0.5x random)
        """
        if start_multiplier is None:
            start_multiplier = random.uniform(1.5, 3.0)
        if end_multiplier is None:
            end_multiplier = random.uniform(0.2, 0.5)
        
        length = len(audio)
        x = np.linspace(0, 4, length)
        exp_decay = np.exp(-x)
        volume_curve = end_multiplier + (start_multiplier - end_multiplier) * exp_decay
        
        audio *= volume_curve
        return audio
    
    def _apply_volume_fluctuation(self, audio, sample_rate, base_multiplier=None, 
                                  fluctuation_depth=None, frequency=None):
        """
        Apply volume fluctuation pattern.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            base_multiplier: Base volume multiplier
            fluctuation_depth: Fluctuation depth (default 0.5-1.5x random)
            frequency: Fluctuation frequency (default 0.5-1.5Hz random)
        """
        if base_multiplier is None:
            base_multiplier = 1
        if fluctuation_depth is None:
            fluctuation_depth = random.uniform(0.5, 1.5)
        if frequency is None:
            frequency = random.uniform(0.5, 1.5)
        
        length = len(audio)
        duration = length / sample_rate
        
        t = np.linspace(0, 2 * np.pi * frequency * duration, length)
        sine_wave = np.sin(t)
        
        volume_pattern = base_multiplier + fluctuation_depth * sine_wave
        volume_pattern = np.maximum(volume_pattern, 0.1)
        
        audio *= volume_pattern
        return audio
