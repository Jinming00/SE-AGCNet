"""
Validation utilities for model evaluation during training.
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import torch

from metrics import calculate_pesq_batch


def _score_dnsmos(dnsmos_scorer, audio, sampling_rate, label, skipped_count):
    if dnsmos_scorer is None:
        return None, skipped_count

    try:
        result = dnsmos_scorer.score_waveform(audio, sampling_rate=int(sampling_rate))
        score = float(result.get('OVRL'))
    except Exception as e:
        skipped_count += 1
        if skipped_count <= 3:
            print(f"[validation] DNSMOS failed for {label}: {e}")
        return None, skipped_count

    if not np.isfinite(score):
        skipped_count += 1
        return None, skipped_count
    return score, skipped_count


def validate_using_inference(model, h, validset, device, batch_size=96, dnsmos_scorer=None,
                             return_metrics=False, label='validation'):
    """
    Validate model using inference pipeline to ensure consistency.
    
    Args:
        model: The model to validate
        h: Hyperparameters
        validset: Validation dataset
        device: Computing device
        batch_size: Batch size for processing (default: 96)
        
    Returns:
        Average PESQ score, or a metrics dict when return_metrics=True.
    """
    try:
        import importlib
        inference_module = importlib.import_module('inference')
        inference_single_file = inference_module.inference_single_file
        
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced_dir = os.path.join(temp_dir, 'enhanced')
            os.makedirs(enhanced_dir, exist_ok=True)
            
            export_records = []
            total_files = len(validset.audio_indexes)
            total_batches = (total_files - 1) // batch_size + 1
            print(f"{label}: running inference on {total_files} files (batch size: {batch_size}, {total_batches} batches)...")
            
            with torch.no_grad():
                for batch_start in range(0, total_files, batch_size):
                    batch_end = min(batch_start + batch_size, total_files)
                    batch_num = batch_start // batch_size + 1
                    
                    print(f"Processing batch {batch_num}/{total_batches} ({batch_start+1}-{batch_end} files)...")
                    
                    for i in range(batch_start, batch_end):
                        try:
                            filename = validset.audio_indexes[i]
                            
                            # Load audio files
                            clean_file = validset._find_file_in_dirs(filename, validset.clean_wavs_dirs)
                            noisy_file = validset._find_file_in_dirs(filename, validset.noisy_wavs_dirs)
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
                                export_records.append({
                                    'filename': filename,
                                    'clean_file': clean_file,
                                    'enhanced_file': enhanced_file,
                                })
                        
                        except Exception as e:
                            print(f"Error processing validation file {i} ({filename}): {e}")
                            continue

                    print(f"Batch {batch_num} inference completed: {len(export_records)} total files exported")
                    
                    # Clean up memory
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    import gc
                    gc.collect()

            print(f"{label}: inference finished, calculating PESQ and DNSMOS metrics...")
            all_file_pesq_scores = []
            all_dnsmos_scores = []
            dnsmos_skipped = 0

            for batch_start in range(0, len(export_records), batch_size):
                batch_end = min(batch_start + batch_size, len(export_records))
                batch_records = export_records[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1

                batch_enhanced_audios = []
                batch_clean_audios = []

                for record in batch_records:
                    try:
                        enhanced_audio, _ = sf.read(record['enhanced_file'])
                        clean_audio, _ = sf.read(record['clean_file'])

                        min_len = min(len(enhanced_audio), len(clean_audio))
                        enhanced_audio = enhanced_audio[:min_len]
                        clean_audio = clean_audio[:min_len]

                        dnsmos_score, dnsmos_skipped = _score_dnsmos(
                            dnsmos_scorer,
                            enhanced_audio,
                            h.sampling_rate,
                            record['filename'],
                            dnsmos_skipped,
                        )
                        if dnsmos_score is not None:
                            all_dnsmos_scores.append(dnsmos_score)

                        batch_enhanced_audios.append(enhanced_audio)
                        batch_clean_audios.append(clean_audio)
                    except Exception as e:
                        print(f"Error evaluating validation file {record.get('filename')}: {e}")

                if len(batch_enhanced_audios) == 0:
                    print(f"Metric batch {batch_num}: no valid files")
                    continue

                batch_pesq_scores = calculate_pesq_batch(
                    batch_clean_audios,
                    batch_enhanced_audios,
                    h.sampling_rate,
                    n_jobs=min(32, len(batch_enhanced_audios)),
                )
                if batch_pesq_scores:
                    all_file_pesq_scores.extend(batch_pesq_scores)
                    batch_mean = np.mean(batch_pesq_scores)
                    print(f"Metric batch {batch_num} PESQ: {batch_mean:.4f} (from {len(batch_pesq_scores)} files)")
                else:
                    print(f"Metric batch {batch_num} PESQ calculation failed")
            
            # Calculate overall validation score
            if len(all_file_pesq_scores) > 0:
                val_pesq = np.mean(all_file_pesq_scores)
                print(f"\nValidation completed:")
                print(f"Total files processed: {len(all_file_pesq_scores)}")
                print(f"PESQ score range: {np.min(all_file_pesq_scores):.4f} - {np.max(all_file_pesq_scores):.4f}")
                print(f"Overall Validation PESQ: {val_pesq:.4f}")
                metrics = {
                    'pesq': float(val_pesq),
                    'pesq_count': int(len(all_file_pesq_scores)),
                    'validation_size': int(total_files),
                    'inference_count': int(len(export_records)),
                }
                if dnsmos_scorer is not None:
                    avg_dnsmos = np.mean(all_dnsmos_scores) if len(all_dnsmos_scores) > 0 else float('nan')
                    print(
                        f"Overall Validation DNSMOS OVRL: {avg_dnsmos:.4f} "
                        f"(count={len(all_dnsmos_scores)}, skipped={dnsmos_skipped})"
                    )
                    metrics.update({
                        'dnsmos_ovrl': float(avg_dnsmos),
                        'dnsmos_count': int(len(all_dnsmos_scores)),
                        'dnsmos_skipped': int(dnsmos_skipped),
                    })
                return metrics if return_metrics else val_pesq
            else:
                print("No valid PESQ scores from any file")
                return None
    
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None
