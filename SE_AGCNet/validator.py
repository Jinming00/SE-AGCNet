"""
Validation utilities for model evaluation during training.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import torch

from metrics import calculate_pesq_batch


def validate_using_inference(model, h, validset, device, batch_size=96):
    """
    Validate model using inference pipeline to ensure consistency.
    
    Args:
        model: The model to validate
        h: Hyperparameters
        validset: Validation dataset
        device: Computing device
        batch_size: Batch size for processing (default: 96)
        
    Returns:
        Average PESQ score or None if validation fails
    """
    try:
        import importlib
        inference_module = importlib.import_module('inference')
        inference_single_file = inference_module.inference_single_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_dir = os.path.join(temp_dir, 'enhanced')
            os.makedirs(enhanced_dir, exist_ok=True)
            
            all_file_pesq_scores = []
            total_files = len(validset.audio_indexes)
            total_batches = (total_files - 1) // batch_size + 1
            print(f"Validating on all {total_files} files (batch size: {batch_size}, {total_batches} batches)...")
            
            with torch.no_grad():
                for batch_start in range(0, total_files, batch_size):
                    batch_end = min(batch_start + batch_size, total_files)
                    batch_num = batch_start // batch_size + 1
                    
                    print(f"Processing batch {batch_num}/{total_batches} ({batch_start+1}-{batch_end} files)...")
                    
                    # Current batch audio pairs
                    batch_enhanced_audios = []
                    batch_clean_audios = []
                    
                    for i in range(batch_start, batch_end):
                        try:
                            filename = validset.audio_indexes[i]
                            
                            # Load audio files
                            clean_file = os.path.join(validset.clean_wavs_dir, filename + '.wav')
                            noisy_file = os.path.join(validset.noisy_wavs_dir, filename + '.wav')
                            enhanced_file = os.path.join(enhanced_dir, filename + '.wav')
                            
                            # Wrapper for model compatibility
                            class ModelWrapper:
                                def __init__(self, model):
                                    self.model = model
                                
                                def __call__(self, *args, **kwargs):
                                    return self.model(*args, **kwargs)
                                
                                def eval(self):
                                    return self.model.eval()
                            
                            wrapped_model = ModelWrapper(model)
                            
                            # Run inference
                            success = inference_single_file(
                                wrapped_model, h, noisy_file, enhanced_file, device,
                                max_length=32000, batch_size=8, 
                                use_chunk_norm=True, overlap_ratio=0.5
                            )
                            
                            if success:
                                # Load enhanced and clean audio
                                enhanced_audio, _ = sf.read(enhanced_file)
                                clean_audio, _ = sf.read(clean_file)
                                
                                # Align lengths
                                min_len = min(len(enhanced_audio), len(clean_audio))
                                enhanced_audio = enhanced_audio[:min_len]
                                clean_audio = clean_audio[:min_len]
                                
                                batch_enhanced_audios.append(enhanced_audio)
                                batch_clean_audios.append(clean_audio)
                        
                        except Exception as e:
                            print(f"Error processing validation file {i} ({filename}): {e}")
                            continue
                    
                    # Calculate PESQ for current batch
                    if len(batch_enhanced_audios) > 0:
                        print(f"Batch {batch_num} completed: {len(batch_enhanced_audios)} files processed")
                        print(f"Calculating PESQ for batch {batch_num}...")
                        
                        batch_pesq_scores = calculate_pesq_batch(
                            batch_clean_audios, batch_enhanced_audios, 
                            h.sampling_rate, 
                            n_jobs=min(32, len(batch_enhanced_audios))
                        )
                        
                        if batch_pesq_scores:
                            all_file_pesq_scores.extend(batch_pesq_scores)
                            batch_mean = np.mean(batch_pesq_scores)
                            print(f"Batch {batch_num} PESQ: {batch_mean:.4f} (from {len(batch_pesq_scores)} files)")
                        else:
                            print(f"Batch {batch_num} PESQ calculation failed")
                    else:
                        print(f"Batch {batch_num} completed: No valid files processed")
                    
                    # Clean up memory
                    del batch_enhanced_audios, batch_clean_audios
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    import gc
                    gc.collect()
            
            # Calculate overall validation score
            if len(all_file_pesq_scores) > 0:
                val_pesq = np.mean(all_file_pesq_scores)
                print(f"\nValidation completed:")
                print(f"Total files processed: {len(all_file_pesq_scores)}")
                print(f"PESQ score range: {np.min(all_file_pesq_scores):.4f} - {np.max(all_file_pesq_scores):.4f}")
                print(f"Overall Validation PESQ: {val_pesq:.4f}")
                return val_pesq
            else:
                print("No valid PESQ scores from any file")
                return None
    
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None

