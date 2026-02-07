#!/usr/bin/env python3
"""LST Distribution per LCZ Class - Cross-references LST tiles with CONUS LCZ raster"""

import glob
import numpy as np
import rasterio
from rasterio.warp import transform_bounds, Resampling
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path
import json

# Configuration
LST_DIR = Path("/workspace/storage/lst-earthformer/Data/ML/Cities_Tiles")
LCZ_FILE = Path("/workspace/storage/lst-earthformer/CONUS_LCZ.tif")
OUTPUT_DIR = Path("/workspace/storage/lst-earthformer")
N_JOBS = 64
TEMP_MIN, TEMP_MAX = -189, 211
NUM_BINS = TEMP_MAX - TEMP_MIN + 1

LCZ_LABELS = {
    1: "Compact high-rise", 2: "Compact mid-rise", 3: "Compact low-rise",
    4: "Open high-rise", 5: "Open mid-rise", 6: "Open low-rise",
    8: "Large low-rise", 10: "Heavy industry",
    11: "Dense trees", 12: "Scattered trees", 13: "Bush, scrub",
    14: "Low plants", 15: "Bare rock", 16: "Bare soil",
    17: "Water", 18: "Custom/Unknown",
}

# Short labels for plots
LCZ_SHORT = {
    1: "Compact\nhigh", 2: "Compact\nmid", 3: "Compact\nlow",
    4: "Open\nhigh", 5: "Open\nmid", 6: "Open\nlow",
    8: "Large\nlow", 10: "Heavy\nindustry",
    11: "Dense\ntrees", 12: "Scattered\ntrees", 13: "Bush\nscrub",
    14: "Low\nplants", 15: "Bare\nrock", 16: "Bare\nsoil",
    17: "Water", 18: "Custom",
}

MAX_LCZ = 19


def process_file(filepath, lcz_file):
    """Process one LST tile: read LST values, sample LCZ at same locations, return per-class histograms."""
    try:
        with rasterio.open(filepath) as lst_src:
            lst_data = lst_src.read(1)
            lst_bounds = lst_src.bounds
            lst_crs = lst_src.crs
            lst_h, lst_w = lst_data.shape

        # Transform LST bounds to LCZ CRS (4326)
        lcz_bounds = transform_bounds(lst_crs, 'EPSG:4326',
                                      lst_bounds.left, lst_bounds.bottom,
                                      lst_bounds.right, lst_bounds.top)

        with rasterio.open(lcz_file) as lcz_src:
            # Get window in LCZ raster corresponding to LST tile bounds
            try:
                win = window_from_bounds(*lcz_bounds, transform=lcz_src.transform)
            except Exception:
                return np.zeros((MAX_LCZ, NUM_BINS), dtype=np.int64)

            # Read LCZ data for this window
            lcz_data = lcz_src.read(1, window=win)

            if lcz_data.size == 0:
                return np.zeros((MAX_LCZ, NUM_BINS), dtype=np.int64)

            # Resize LCZ to match LST tile (128x128) using nearest neighbor
            from PIL import Image
            lcz_resized = np.array(
                Image.fromarray(lcz_data).resize((lst_w, lst_h), Image.NEAREST)
            )

        # Build per-LCZ-class histograms
        histograms = np.zeros((MAX_LCZ, NUM_BINS), dtype=np.int64)
        bins = np.arange(TEMP_MIN, TEMP_MAX + 2)

        for lcz_class in range(1, MAX_LCZ):
            mask = (lcz_resized == lcz_class) & (lst_data != 0)
            if mask.any():
                vals = lst_data[mask]
                counts, _ = np.histogram(vals, bins=bins)
                histograms[lcz_class] = counts

        return histograms

    except Exception as e:
        return np.zeros((MAX_LCZ, NUM_BINS), dtype=np.int64)


def main():
    print("Finding LST files...")
    files = glob.glob(str(LST_DIR / "**" / "LST_*.tif"), recursive=True)
    total_files = len(files)
    print(f"Found {total_files:,} files")

    if total_files == 0:
        print("No files found.")
        return

    print(f"Processing with {N_JOBS} workers...")
    results = Parallel(n_jobs=N_JOBS, prefer="processes", verbose=10)(
        delayed(process_file)(f, str(LCZ_FILE)) for f in files
    )

    print("Aggregating histograms...")
    total_hists = np.sum(results, axis=0)  # shape: (MAX_LCZ, NUM_BINS)
    temperatures = np.arange(TEMP_MIN, TEMP_MAX + 1)

    # Find which LCZ classes have data
    present_classes = [c for c in range(1, MAX_LCZ) if total_hists[c].sum() > 0]
    print(f"LCZ classes with data: {present_classes}")

    # Compute stats per class
    stats = {}
    for c in present_classes:
        hist = total_hists[c]
        total = hist.sum()
        if total == 0:
            continue
        prob = hist / total
        mean = np.sum(temperatures * hist) / total
        var = np.sum(hist * (temperatures - mean) ** 2) / total
        std = np.sqrt(var)
        cumsum = np.cumsum(hist)
        median_idx = np.searchsorted(cumsum, total * 0.5)
        median = temperatures[min(median_idx, len(temperatures) - 1)]
        nonzero = np.nonzero(hist)[0]
        tmin = temperatures[nonzero[0]]
        tmax = temperatures[nonzero[-1]]
        stats[c] = {
            'total_pixels': int(total),
            'mean': float(mean),
            'std': float(std),
            'median': int(median),
            'min': int(tmin),
            'max': int(tmax),
        }

    # --- Plot 1: Overlay probability distributions ---
    print("Creating overlay plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.tab20
    for i, c in enumerate(present_classes):
        hist = total_hists[c]
        total = hist.sum()
        if total == 0:
            continue
        prob = hist / total
        label = f"LCZ {c}: {LCZ_LABELS.get(c, '?')}"
        color = cmap(i / len(present_classes))
        ax.plot(temperatures, prob, label=label, color=color, linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Temperature (F)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("LST Distribution by LCZ Class", fontsize=14)
    ax.set_xlim(0, 180)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lst_per_lcz_overlay.png", dpi=150)
    plt.close()
    print("Saved: lst_per_lcz_overlay.png")

    # --- Plot 2: Subplots per LCZ class ---
    print("Creating subplot grid...")
    n_classes = len(present_classes)
    ncols = 4
    nrows = (n_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, c in enumerate(present_classes):
        ax = axes[i]
        hist = total_hists[c]
        total = hist.sum()
        prob = hist / total
        s = stats[c]
        ax.bar(temperatures, prob, width=1, color='steelblue', edgecolor='none')
        ax.set_title(f"LCZ {c}: {LCZ_LABELS.get(c, '?')}", fontsize=10)
        ax.set_xlim(0, 180)
        ax.axvline(s['mean'], color='red', linestyle='--', linewidth=0.8, label=f"Mean: {s['mean']:.1f}F")
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        info = f"n={total:,}\nstd={s['std']:.1f}F"
        ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    # Common labels
    fig.text(0.5, 0.02, "Temperature (F)", ha='center', fontsize=12)
    fig.text(0.02, 0.5, "Probability", va='center', rotation='vertical', fontsize=12)
    fig.suptitle("LST Temperature Distribution by LCZ Class", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "lst_per_lcz_subplots.png", dpi=150)
    plt.close()
    print("Saved: lst_per_lcz_subplots.png")

    # --- Plot 3: Box-whisker style summary ---
    print("Creating summary plot...")
    fig, ax = plt.subplots(figsize=(14, 6))
    means = [stats[c]['mean'] for c in present_classes]
    stds = [stats[c]['std'] for c in present_classes]
    labels = [f"LCZ {c}" for c in present_classes]
    x = range(len(present_classes))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LCZ_SHORT.get(c, str(c)) for c in present_classes], fontsize=8)
    ax.set_ylabel("Mean LST (F)", fontsize=12)
    ax.set_title("Mean LST by LCZ Class (error bars = 1 std)", fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    for i, (m, c) in enumerate(zip(means, present_classes)):
        ax.text(i, m + stds[i] + 1, f"{m:.1f}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lst_per_lcz_means.png", dpi=150)
    plt.close()
    print("Saved: lst_per_lcz_means.png")

    # --- Stats file ---
    print("Writing stats...")
    with open(OUTPUT_DIR / "lst_per_lcz_stats.txt", "w") as f:
        f.write("LST Distribution per LCZ Class\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total LST files processed: {total_files:,}\n")
        f.write(f"LCZ classes with data: {len(present_classes)}\n\n")
        f.write(f"{'Class':<8}{'Description':<25}{'Pixels':>15}{'Mean':>10}{'Std':>10}{'Median':>10}{'Min':>8}{'Max':>8}\n")
        f.write("-" * 94 + "\n")
        for c in present_classes:
            s = stats[c]
            desc = LCZ_LABELS.get(c, 'Unknown')
            f.write(f"LCZ {c:<4}{desc:<25}{s['total_pixels']:>15,}{s['mean']:>10.2f}{s['std']:>10.2f}{s['median']:>10}{s['min']:>8}{s['max']:>8}\n")
        f.write("\n")

        total_all = sum(s['total_pixels'] for s in stats.values())
        f.write(f"{'Total':<33}{total_all:>15,}\n")

    print(f"Saved: lst_per_lcz_stats.txt")
    print("Done!")


if __name__ == "__main__":
    main()
