"""
Metadata Generation Module

Generates JSON metadata for audio processing:
- Per-combination details
- Global statistics
- Speaker usage
- Augmentation distribution
"""

import os
import json
import datetime
import numpy as np


class MetadataGenerator:
    """
    Metadata generator for recording audio processing details.
    
    Generates:
    1. Global processing statistics
    2. Per-combination detailed information
    3. JSON format metadata files
    """
    
    def __init__(self, target_dir):
        """
        Initialize metadata generator.
        
        Args:
            target_dir: Target directory for audio processing
        """
        self.target_dir = target_dir
        self.metadata_dir = os.path.join(target_dir, "metadata")
        self.global_stats = {
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_combinations": 0,
            "total_segments": 0,
            "total_duration_hours": 0.0,
            "total_speakers": set(),
            "volume_range": {
                "min": 1.0,
                "max": 1.0
            },
            "augmentation_stats": {
                "none": 0,
                "sudden_spikes": 0,
                "gradual_increase": 0,
                "gradual_decrease": 0,
                "volume_fluctuation": 0
            }
        }
        
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def generate_metadata_files(self, combinations):
        """
        Generate metadata files for processed audio combinations.
        
        Args:
            combinations: List of processed audio combinations
        
        Returns:
            dict: Metadata generation statistics
        """
        combo_metadata_dir = os.path.join(self.metadata_dir, "combinations")
        os.makedirs(combo_metadata_dir, exist_ok=True)
        
        self.global_stats["total_combinations"] = len(combinations)
        
        all_volumes = []
        
        for combo in combinations:
            try:
                combo_id = combo['combo_id']
                
                combo_metadata = {
                    "combo_id": combo_id,
                    "speakers": combo['speakers'],
                    "total_duration": combo['total_duration'],
                    "num_segments": combo['num_segments'],
                    "origin_file": os.path.basename(combo['origin_output_file']),
                    "lower_file": os.path.basename(combo['lower_output_file']),
                    "segments": []
                }
                
                for segment in combo['file_info']:
                    segment_info = {
                        "speaker_id": segment['speaker_id'],
                        "duration": segment['duration'],
                        "volume_factor": segment['volume'],
                        "augmentation": {
                            "mode": segment['augmentation_mode'],
                            "description": segment['augmentation_description']
                        },
                        "original_file": os.path.basename(segment['original_file'])
                    }
                    combo_metadata["segments"].append(segment_info)
                    
                    self.global_stats["total_segments"] += 1
                    self.global_stats["total_speakers"].add(segment['speaker_id'])
                    self.global_stats["augmentation_stats"][segment['augmentation_mode']] += 1
                    all_volumes.append(segment['volume'])
                
                combo_filename = f"combo_{combo_id:04d}_metadata.json"
                combo_metadata_path = os.path.join(combo_metadata_dir, combo_filename)
                
                with open(combo_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(combo_metadata, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                continue
        
        self.global_stats["total_duration_hours"] = sum(combo["total_duration"] for combo in combinations) / 3600
        self.global_stats["total_speakers"] = list(self.global_stats["total_speakers"])
        
        if all_volumes:
            self.global_stats["volume_range"]["min"] = min(all_volumes)
            self.global_stats["volume_range"]["max"] = max(all_volumes)
        
        total_segments = self.global_stats["total_segments"]
        if total_segments > 0:
            mode_percentages = {}
            for mode in list(self.global_stats["augmentation_stats"].keys()):
                count = self.global_stats["augmentation_stats"][mode]
                percentage = round(count / total_segments * 100, 2)
                mode_percentages[f"{mode}_percent"] = percentage
            
            self.global_stats["augmentation_stats"].update(mode_percentages)
        
        global_stats_path = os.path.join(self.metadata_dir, "global_stats.json")
        
        stats_for_json = self.global_stats.copy()
        
        with open(global_stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
        
        dataset_description = {
            "dataset_name": os.path.basename(self.target_dir),
            "creation_date": self.global_stats["processing_date"],
            "total_combinations": self.global_stats["total_combinations"],
            "total_segments": self.global_stats["total_segments"],
            "total_duration_hours": self.global_stats["total_duration_hours"],
            "num_speakers": len(self.global_stats["total_speakers"]),
            "volume_range": self.global_stats["volume_range"],
            "augmentation_summary": {
                "none_percent": self.global_stats["augmentation_stats"].get("none_percent", 0),
                "augmented_percent": 100 - self.global_stats["augmentation_stats"].get("none_percent", 0),
                "modes_used": [
                    mode for mode in ["sudden_spikes", "gradual_increase", "gradual_decrease", "volume_fluctuation"]
                    if self.global_stats["augmentation_stats"].get(f"{mode}_percent", 0) > 0
                ]
            },
            "directory_structure": {
                "origin": "Original or peak normalized audio files",
                "lower": "Volume-adjusted audio files",
                "transcriptions": "Text transcription files",
                "rttm": "Rich Transcription Time Marked files",
                "ctm": "Conversation Time Marked files",
                "metadata": "Metadata and processing statistics"
            }
        }
        
        dataset_path = os.path.join(self.metadata_dir, "dataset_description.json")
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_description, f, indent=2, ensure_ascii=False)
        
        processing_summary = {
            "processing_date": self.global_stats["processing_date"],
            "dataset_summary": {
                "total_combinations": self.global_stats["total_combinations"],
                "total_speakers": len(self.global_stats["total_speakers"]),
                "total_segments": self.global_stats["total_segments"],
                "total_duration_hours": round(self.global_stats["total_duration_hours"], 2)
            },
            "volume_modification": {
                "range": f"{self.global_stats['volume_range']['min']*100:.1f}% - {self.global_stats['volume_range']['max']*100:.1f}%",
                "description": "Volume percentage range relative to original volume"
            },
            "augmentation_stats": {
                "none": {
                    "count": self.global_stats["augmentation_stats"]["none"],
                    "percent": self.global_stats["augmentation_stats"].get("none_percent", 0)
                },
                "sudden_spikes": {
                    "count": self.global_stats["augmentation_stats"]["sudden_spikes"],
                    "percent": self.global_stats["augmentation_stats"].get("sudden_spikes_percent", 0),
                    "description": "Sudden volume spikes - random short-duration volume peaks"
                },
                "gradual_increase": {
                    "count": self.global_stats["augmentation_stats"]["gradual_increase"],
                    "percent": self.global_stats["augmentation_stats"].get("gradual_increase_percent", 0),
                    "description": "Gradual increase - progressive volume increase from low to high"
                },
                "gradual_decrease": {
                    "count": self.global_stats["augmentation_stats"]["gradual_decrease"],
                    "percent": self.global_stats["augmentation_stats"].get("gradual_decrease_percent", 0),
                    "description": "Gradual decrease - progressive volume decrease from high to low"
                },
                "volume_fluctuation": {
                    "count": self.global_stats["augmentation_stats"]["volume_fluctuation"],
                    "percent": self.global_stats["augmentation_stats"].get("volume_fluctuation_percent", 0),
                    "description": "Volume fluctuation - wave-like volume variations"
                }
            }
        }
        
        summary_path = os.path.join(self.metadata_dir, "processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, ensure_ascii=False)
        
        return {
            "metadata_dir": self.metadata_dir,
            "total_combinations": self.global_stats["total_combinations"],
            "total_segments": self.global_stats["total_segments"],
            "total_duration_hours": self.global_stats["total_duration_hours"]
        }


def generate_metadata_files(combinations, target_dir):
    """
    Generate metadata files for processed audio combinations (convenience function).
    
    Args:
        combinations: List of processed audio combinations
        target_dir: Target directory
    
    Returns:
        dict: Metadata generation statistics
    """
    metadata_generator = MetadataGenerator(target_dir)
    return metadata_generator.generate_metadata_files(combinations)
