#!/usr/bin/env python3
"""Regenerate LST-per-LCZ subplots from histogram data with percentage y-axis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "out"

LCZ_LABELS = {
    1: "Compact high-rise", 2: "Compact mid-rise", 3: "Compact low-rise",
    4: "Open high-rise", 5: "Open mid-rise", 6: "Open low-rise",
    8: "Large low-rise", 10: "Heavy industry",
    11: "Dense trees", 12: "Scattered trees", 13: "Bush, scrub",
    14: "Low plants", 15: "Bare rock", 16: "Bare soil",
    17: "Water", 18: "Custom/Unknown",
}

# Parse histogram file
data = {}  # {class_id: {'total': int, 'temps': [], 'counts': []}}
current_class = None

with open(OUTPUT_DIR / "lst_per_lcz_histograms.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("LCZ_CLASS"):
            parts = line.split()
            current_class = int(parts[1])
            total = int(parts[-1].split("=")[1])
            data[current_class] = {"total": total, "temps": [], "counts": []}
        elif current_class and line and not line.startswith("Temp") and "\t" in line:
            parts = line.split("\t")
            data[current_class]["temps"].append(int(parts[0]))
            data[current_class]["counts"].append(int(parts[1]))

present_classes = sorted(data.keys())

# Compute stats
stats = {}
for c in present_classes:
    temps = np.array(data[c]["temps"])
    counts = np.array(data[c]["counts"], dtype=np.float64)
    total = counts.sum()
    mean = np.sum(temps * counts) / total
    var = np.sum(counts * (temps - mean) ** 2) / total
    stats[c] = {"mean": mean, "std": np.sqrt(var), "total": int(total)}

# Build full arrays for plotting (fill gaps with 0)
TEMP_MIN, TEMP_MAX = -189, 211
all_temps = np.arange(TEMP_MIN, TEMP_MAX + 1)
histograms = {}
for c in present_classes:
    hist = np.zeros(len(all_temps))
    temp_to_idx = {t: i for i, t in enumerate(all_temps)}
    for t, cnt in zip(data[c]["temps"], data[c]["counts"]):
        if t in temp_to_idx:
            hist[temp_to_idx[t]] = cnt
    total = hist.sum()
    histograms[c] = (hist / total) * 100  # convert to percentage

# Plot
ncols = 4
nrows = (len(present_classes) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), sharex=True, sharey=False)
axes = axes.flatten()

pct_fmt = FuncFormatter(lambda x, _: f"{x:.1f}%")

for i, c in enumerate(present_classes):
    ax = axes[i]
    pct = histograms[c]
    s = stats[c]
    ax.bar(all_temps, pct, width=1, color='steelblue', edgecolor='none')
    ax.set_title(f"LCZ {c}: {LCZ_LABELS.get(c, '?')}", fontsize=10)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.axvline(s['mean'], color='red', linestyle='--', linewidth=0.8, label=f"Mean: {s['mean']:.1f}F")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    info = f"n={s['total']:,}\nstd={s['std']:.1f}F"
    ax.text(0.97, 0.95, info, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

for i in range(len(present_classes), len(axes)):
    axes[i].set_visible(False)

fig.text(0.5, 0.02, "Temperature (F)", ha='center', fontsize=12)
fig.text(0.02, 0.5, "Percentage", va='center', rotation='vertical', fontsize=12)
fig.suptitle("LST Temperature Distribution by LCZ Class", fontsize=14, y=0.98)
plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
plt.savefig(OUTPUT_DIR / "lst_per_lcz_subplots.png", dpi=150)
plt.close()
print("Saved: lst_per_lcz_subplots.png")
