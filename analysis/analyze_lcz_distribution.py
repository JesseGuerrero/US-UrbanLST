#!/usr/bin/env python3
"""LCZ Class Distribution Analysis - Bar Chart"""

import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path

# Configuration
LCZ_FILE = Path("/workspace/storage/lst-earthformer/CONUS_LCZ.tif")
OUTPUT_DIR = Path("/workspace/storage/lst-earthformer")
N_JOBS = 124
MAX_CLASS = 19

LCZ_LABELS = {
    1: "Compact high-rise",
    2: "Compact mid-rise",
    3: "Compact low-rise",
    4: "Open high-rise",
    5: "Open mid-rise",
    6: "Open low-rise",
    8: "Large low-rise",
    10: "Heavy industry",
    11: "Dense trees",
    12: "Scattered trees",
    13: "Bush, scrub",
    14: "Low plants",
    15: "Bare rock",
    16: "Bare soil",
    17: "Water",
    18: "Custom/Unknown",
}

def process_chunk(lcz_file, y_start, chunk_height, width):
    """Process a horizontal strip of the raster."""
    try:
        with rasterio.open(lcz_file) as src:
            window = Window(0, y_start, width, chunk_height)
            data = src.read(1, window=window).flatten()
            data = data[data != 0]  # Exclude class 0 (no data)
            counts = np.bincount(data.astype(np.int64), minlength=MAX_CLASS)
            return counts
    except Exception as e:
        print(f"Error processing chunk at row {y_start}: {e}")
        return np.zeros(MAX_CLASS, dtype=np.int64)

def main():
    print("Opening LCZ raster...")
    with rasterio.open(LCZ_FILE) as src:
        height, width = src.height, src.width
    print(f"Dimensions: {width:,} x {height:,} pixels ({width * height:,} total)")

    # Divide into chunks
    chunk_size = height // N_JOBS
    chunks = []
    for i in range(N_JOBS):
        y_start = i * chunk_size
        chunk_height = chunk_size if i < N_JOBS - 1 else height - y_start
        chunks.append((y_start, chunk_height))

    print(f"Processing {len(chunks)} chunks with {N_JOBS} workers...")
    counts_list = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=10)(
        delayed(process_chunk)(str(LCZ_FILE), y, h, width) for y, h in chunks
    )

    print("Aggregating counts...")
    total_counts = np.sum(counts_list, axis=0)
    total_pixels = total_counts.sum()

    # Filter to classes that exist (excluding class 0)
    present_classes = [i for i in range(1, MAX_CLASS) if total_counts[i] > 0]
    counts = [total_counts[i] for i in present_classes]
    percentages = [c / total_pixels * 100 for c in counts]

    # Dominant class
    dominant_idx = np.argmax(counts)
    dominant_class = present_classes[dominant_idx]

    # Ranking by coverage
    ranking = sorted(zip(present_classes, counts, percentages), key=lambda x: -x[1])

    # Plot
    print("Creating plot...")
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(present_classes)), counts, color="forestgreen", edgecolor="none")
    plt.xticks(range(len(present_classes)), [str(c) for c in present_classes], fontsize=10)
    plt.xlabel("LCZ Class", fontsize=12)
    plt.ylabel("Pixel Count", fontsize=12)
    plt.title(f"LCZ Class Distribution ({total_pixels:,} pixels, excluding class 0)", fontsize=14)
    plt.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars, percentages):
        if pct >= 1:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lcz_class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved: lcz_class_distribution.png")

    # Statistics file
    with open(OUTPUT_DIR / "lcz_class_stats.txt", "w") as f:
        f.write("LCZ Class Distribution Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Image dimensions: {width:,} x {height:,}\n")
        f.write(f"Total pixels (excluding class 0): {total_pixels:,}\n\n")
        f.write(f"Dominant class: {dominant_class} ({LCZ_LABELS.get(dominant_class, 'Unknown')})\n\n")
        f.write("Class Ranking by Area Coverage:\n")
        f.write("-" * 60 + "\n")
        f.write("Rank  Class       Count    Percentage  Description\n")
        f.write("-" * 60 + "\n")
        for rank, (cls, cnt, pct) in enumerate(ranking, 1):
            desc = LCZ_LABELS.get(cls, "Unknown")
            f.write(f"{rank:<6}{cls:<8}{cnt:>15,}{pct:>11.4f}%  {desc}\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"Saved: lcz_class_stats.txt")

if __name__ == "__main__":
    main()
