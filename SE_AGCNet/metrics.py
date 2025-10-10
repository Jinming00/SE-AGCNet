"""
Evaluation metrics for audio quality assessment.
"""

import numpy as np
from pesq import pesq
from multiprocessing import Pool
from functools import partial


def calculate_pesq_single(audio_pair, sample_rate=16000):
    """
    Calculate PESQ score for a single audio pair.
    
    Args:
        audio_pair: Tuple of (reference_audio, degraded_audio)
        sample_rate: Sampling rate in Hz (default: 16000)
        
    Returns:
        PESQ score or None if calculation fails
    """
    ref_audio, deg_audio = audio_pair
    try:
        # Convert to mono if stereo
        if len(ref_audio.shape) > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if len(deg_audio.shape) > 1:
            deg_audio = np.mean(deg_audio, axis=1)
        
        # Align lengths
        min_len = min(len(ref_audio), len(deg_audio))
        ref_audio = ref_audio[:min_len]
        deg_audio = deg_audio[:min_len]
        
        # Calculate PESQ score
        pesq_score = pesq(sample_rate, ref_audio, deg_audio, 'wb')
        return pesq_score
    except Exception as e:
        print(f"PESQ calculation error: {e}")
        return None


def calculate_pesq_batch(ref_audios, deg_audios, sample_rate=16000, n_jobs=8):
    """
    Calculate PESQ scores for multiple audio pairs in parallel.
    
    Args:
        ref_audios: List of reference audio arrays
        deg_audios: List of degraded audio arrays
        sample_rate: Sampling rate in Hz (default: 16000)
        n_jobs: Number of parallel processes (default: 8)
        
    Returns:
        List of valid PESQ scores
    """
    print(f"Calculating PESQ for {len(ref_audios)} audio pairs using {n_jobs} processes...")
    
    audio_pairs = list(zip(ref_audios, deg_audios))
    calc_func = partial(calculate_pesq_single, sample_rate=sample_rate)
    
    with Pool(n_jobs) as pool:
        results = pool.map(calc_func, audio_pairs)
    
    # Filter out failed calculations
    valid_scores = [score for score in results if score is not None]
    failed_count = len(results) - len(valid_scores)
    
    print(f"PESQ calculation completed: {len(valid_scores)}/{len(results)} successful")
    if failed_count > 0:
        print(f"Warning: {failed_count} PESQ calculations failed")
    
    return valid_scores



