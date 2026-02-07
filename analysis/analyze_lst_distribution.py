#!/usr/bin/env python3
"""LST Temperature Distribution Analysis - Probability Histogram"""

import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path

# Configuration
LST_DIR = Path("/workspace/storage/lst-earthformer/Data/ML/Cities_Tiles")
OUTPUT_DIR = Path("/workspace/storage/lst-earthformer")
N_JOBS = 124
TEMP_MIN, TEMP_MAX = -189, 211
BINS = range(TEMP_MIN, TEMP_MAX + 2)

def process_file(filepath):
    """Process single TIF file, return histogram counts."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).flatten()
            data = data[data != 0]  # Exclude 0F (no data)
            counts, _ = np.histogram(data, bins=BINS)
            return counts
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return np.zeros(len(BINS) - 1, dtype=np.int64)

def main():
    print("Finding LST files...")
    files = glob.glob(str(LST_DIR / "**" / "LST_*.tif"), recursive=True)
    total_files = len(files)
    print(f"Found {total_files:,} files")

    if total_files == 0:
        print("No files found. Exiting.")
        return

    print(f"Processing with {N_JOBS} workers...")
    histograms = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=10)(
        delayed(process_file)(f) for f in files
    )

    print("Aggregating histograms...")
    total_hist = np.sum(histograms, axis=0)
    temperatures = np.arange(TEMP_MIN, TEMP_MAX + 1)
    total_pixels = total_hist.sum()

    # Probability distribution
    probabilities = total_hist / total_pixels

    # Statistics
    weighted_temps = temperatures * total_hist
    mean_temp = weighted_temps.sum() / total_pixels
    mode_idx = np.argmax(total_hist)
    mode_temp = temperatures[mode_idx]

    # Cumulative for percentiles
    cumsum = np.cumsum(total_hist)
    percentiles = {}
    for p in [10, 25, 50, 75, 90]:
        idx = np.searchsorted(cumsum, total_pixels * p / 100)
        percentiles[p] = temperatures[min(idx, len(temperatures) - 1)]

    # Find actual min/max with data
    nonzero = np.nonzero(total_hist)[0]
    actual_min = temperatures[nonzero[0]] if len(nonzero) > 0 else TEMP_MIN
    actual_max = temperatures[nonzero[-1]] if len(nonzero) > 0 else TEMP_MAX

    # Variance for std
    variance = np.sum(total_hist * (temperatures - mean_temp) ** 2) / total_pixels
    std_temp = np.sqrt(variance)

    # Plot
    print("Creating plot...")
    plt.figure(figsize=(14, 8))
    plt.bar(temperatures, probabilities, width=1, edgecolor="none", color="steelblue")
    plt.xlabel("Temperature (°F)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title(f"LST Temperature Distribution ({total_files:,} files, {total_pixels:,} pixels, excluding 0°F)", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lst_temperature_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: lst_temperature_distribution.png")

    # Statistics file
    with open(OUTPUT_DIR / "lst_temperature_stats.txt", "w") as f:
        f.write("LST Temperature Distribution Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total files processed: {total_files:,}\n")
        f.write(f"Total pixels (excluding 0°F): {total_pixels:,}\n\n")
        f.write(f"Min temperature: {actual_min}°F\n")
        f.write(f"Max temperature: {actual_max}°F\n")
        f.write(f"Mean temperature: {mean_temp:.2f}°F\n")
        f.write(f"Median temperature: {percentiles[50]}°F\n")
        f.write(f"Mode temperature: {mode_temp}°F\n")
        f.write(f"Std deviation: {std_temp:.2f}°F\n\n")
        f.write("Percentiles:\n")
        for p, t in percentiles.items():
            f.write(f"  {p}th: {t}°F\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write("Full Histogram Data\n")
        f.write("Temp(°F)\tCount\t\tProbability\n")
        f.write("-" * 50 + "\n")
        for i, t in enumerate(temperatures):
            c = total_hist[i]
            p = probabilities[i]
            if c > 0:
                f.write(f"{t}\t\t{c:,}\t\t{p:.8f}\n")
    print(f"Saved: lst_temperature_stats.txt")

if __name__ == "__main__":
    main()
