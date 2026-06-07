"""
Add DNS noise to generated lower audio at random SNR.

This script reads wav files from an existing lower/ directory and writes noisy
copies to a separate output directory. It does not overwrite the original lower
audio.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


DEFAULT_NOISE_DIR = "/projects_vol/gp_aseschng/jinming/data/NOISE/DNS_for_SE-AGCNet"


def list_wav_files(directory):
    return sorted(glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True))


def rms_power(audio):
    return float(np.mean(np.square(audio, dtype=np.float64)))


def mix_at_snr(clean_audio, noise_audio, snr_db, eps=1e-12):
    clean_power = rms_power(clean_audio)
    noise_power = rms_power(noise_audio)

    if clean_power < eps or noise_power < eps:
        return clean_audio.copy(), 0.0

    target_noise_power = clean_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / (noise_power + eps))
    noisy_audio = clean_audio + noise_audio * scale
    return noisy_audio.astype(np.float32), float(scale)


def output_path_for(input_file, lower_dir, output_dir):
    rel_path = os.path.relpath(input_file, lower_dir)
    return os.path.join(output_dir, rel_path)


def process_one_file(task):
    (
        index,
        lower_file,
        lower_dir,
        noise_files,
        output_dir,
        min_snr,
        max_snr,
        sample_rate,
        seed,
    ) = task

    rng = random.Random(seed + index)
    noise_file = rng.choice(noise_files)
    snr_db = rng.uniform(min_snr, max_snr)

    clean_audio, _ = librosa.load(lower_file, sr=sample_rate, mono=True)
    noise_audio, _ = librosa.load(noise_file, sr=sample_rate, mono=True)

    if len(noise_audio) == 0:
        return None

    if len(noise_audio) == len(clean_audio):
        aligned_noise = noise_audio
    elif len(noise_audio) > len(clean_audio):
        max_start = len(noise_audio) - len(clean_audio)
        start = rng.randint(0, max_start)
        aligned_noise = noise_audio[start:start + len(clean_audio)]
    else:
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        aligned_noise = np.tile(noise_audio, repeats)[:len(clean_audio)]

    noisy_audio, noise_scale = mix_at_snr(clean_audio, aligned_noise, snr_db)

    out_file = output_path_for(lower_file, lower_dir, output_dir)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    sf.write(out_file, noisy_audio, sample_rate)

    return {
        "input_file": lower_file,
        "output_file": out_file,
        "noise_file": noise_file,
        "snr_db": f"{snr_db:.4f}",
        "noise_scale": f"{noise_scale:.8f}",
        "sample_rate": sample_rate,
    }


def add_noise_to_lower(
    lower_dir,
    noise_dir,
    output_dir,
    min_snr,
    max_snr,
    sample_rate,
    seed,
    num_workers,
):
    lower_files = list_wav_files(lower_dir)
    noise_files = list_wav_files(noise_dir)

    if not lower_files:
        raise RuntimeError(f"No wav files found in lower_dir: {lower_dir}")
    if not noise_files:
        raise RuntimeError(f"No wav files found in noise_dir: {noise_dir}")

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "noise_metadata.csv")

    tasks = [
        (
            index,
            lower_file,
            lower_dir,
            noise_files,
            output_dir,
            min_snr,
            max_snr,
            sample_rate,
            seed,
        )
        for index, lower_file in enumerate(lower_files)
    ]

    rows = []
    if num_workers <= 1:
        iterator = (process_one_file(task) for task in tasks)
        for row in tqdm(iterator, total=len(tasks), desc="Adding noise", unit="file", ncols=80):
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_one_file, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Adding noise", unit="file", ncols=80):
                row = future.result()
                if row is not None:
                    rows.append(row)

    rows.sort(key=lambda row: row["input_file"])

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_file",
                "output_file",
                "noise_file",
                "snr_db",
                "noise_scale",
                "sample_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add random DNS noise to wav files in a lower/ directory."
    )
    parser.add_argument(
        "--lower_dir",
        required=True,
        help="Input lower directory containing wav files.",
    )
    parser.add_argument(
        "--noise_dir",
        default=DEFAULT_NOISE_DIR,
        help=f"Noise wav directory. Default: {DEFAULT_NOISE_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: sibling directory named lower_noisy.",
    )
    parser.add_argument("--min_snr", type=float, default=5.0)
    parser.add_argument("--max_snr", type=float, default=25.0)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for serial processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lower_dir = os.path.abspath(args.lower_dir)
    noise_dir = os.path.abspath(args.noise_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(lower_dir), "lower_noisy")
    output_dir = os.path.abspath(output_dir)

    if args.min_snr > args.max_snr:
        raise ValueError("--min_snr must be <= --max_snr")

    print(f"Input lower: {lower_dir}")
    print(f"Noise dir:   {noise_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"SNR range:   {args.min_snr}-{args.max_snr} dB")
    print(f"Workers:     {args.num_workers}")

    count, metadata_path = add_noise_to_lower(
        lower_dir=lower_dir,
        noise_dir=noise_dir,
        output_dir=output_dir,
        min_snr=args.min_snr,
        max_snr=args.max_snr,
        sample_rate=args.sample_rate,
        seed=args.seed,
        num_workers=max(1, args.num_workers),
    )

    print(f"Created {count} noisy wav files")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
