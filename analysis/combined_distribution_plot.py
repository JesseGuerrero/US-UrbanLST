#!/usr/bin/env python3
"""Combined LCZ + LST distribution plot in academic style."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.interpolate import PchipInterpolator, make_interp_spline

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

def human_format(x, pos):
    if x >= 1e9:
        return f'{x/1e9:g}B'
    elif x >= 1e6:
        return f'{x/1e6:g}M'
    elif x >= 1e3:
        return f'{x/1e3:g}K'
    elif x >= 1:
        return f'{x:g}'
    return '0'

# ── Parse LST stats ──────────────────────────────────────────────
lst_temps, lst_counts = [], []
in_data = False
OUTPUT_DIR = Path(__file__).resolve().parent / "out"

with open(OUTPUT_DIR / 'lst_temperature_stats.txt') as f:
    for line in f:
        if line.startswith('Temp'):
            in_data = True
            continue
        if line.startswith('---'):
            continue
        if in_data and line.strip():
            parts = line.split()
            lst_temps.append(int(parts[0]))
            lst_counts.append(int(parts[1].replace(',', '')))

lst_temps = np.array(lst_temps)
lst_counts = np.array(lst_counts, dtype=np.float64)

mask = (lst_temps >= -50) & (lst_temps <= 175)
lst_temps_trim = lst_temps[mask]
lst_counts_trim = lst_counts[mask]

bin_width = 5
bin_edges = np.arange(-50, 176, bin_width)
bin_centers = bin_edges[:-1] + bin_width / 2
binned_counts = np.zeros(len(bin_centers), dtype=np.float64)
for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    sel = (lst_temps_trim >= lo) & (lst_temps_trim < hi)
    binned_counts[i] = lst_counts_trim[sel].sum()

# ── Parse LCZ stats ─────────────────────────────────────────────
lcz_classes, lcz_counts = [], []
in_data = False
with open(OUTPUT_DIR / 'lcz_class_stats.txt') as f:
    for line in f:
        if line.startswith('Rank'):
            in_data = True
            continue
        if line.startswith('---') or line.startswith('='):
            if in_data and line.startswith('='):
                in_data = False
            continue
        if in_data and line.strip():
            parts = line.split()
            lcz_classes.append(int(parts[1]))
            lcz_counts.append(int(parts[2].replace(',', '')))

order = np.argsort(lcz_classes)
lcz_classes = np.array(lcz_classes)[order]
lcz_counts = np.array(lcz_counts, dtype=np.float64)[order]
total_lcz = lcz_counts.sum()

# ── Figure ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0))
fig.subplots_adjust(wspace=0.22, left=0.08, right=0.99, top=0.97, bottom=0.12)

bar_color = '#6BAED6'
curve_color = '#2171B5'
fmt = FuncFormatter(human_format)

# ── Left: LCZ class distribution ────────────────────────────────
x_pos = np.arange(len(lcz_classes))
ax1.bar(x_pos, lcz_counts, width=0.75, color=bar_color, edgecolor='white', linewidth=0.3)

# PCHIP: touches every bar top, smooth, no overshoot
pchip = PchipInterpolator(x_pos, lcz_counts)
x_fine = np.linspace(x_pos[0], x_pos[-1], 300)
y_fine = np.maximum(pchip(x_fine), 0)
ax1.plot(x_fine, y_fine, color=curve_color, linewidth=1.0)

ax1.yaxis.set_major_locator(FixedLocator([0, 1.2e6, 2.4e6]))
ax1.yaxis.set_major_formatter(fmt)
ax1.set_ylim(0, 3e6)
ax1.set_xticks(x_pos)
ax1.set_xticklabels([str(c) for c in lcz_classes], fontsize=16)
ax1.set_xlabel('LCZ Class', fontsize=16)
ax1.set_ylabel('Pixels', fontsize=16)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_xlim(-0.6, len(lcz_classes) - 0.4)

dominant_cls = lcz_classes[np.argmax(lcz_counts)]
dominant_pct = lcz_counts.max() / total_lcz * 100
txt = f'Total pixels: ~{int(round(total_lcz, -6) // 1e6):.0f}M\nDominant: Class {dominant_cls} ({dominant_pct:.1f}%)'
ax1.text(0.97, 0.97, txt, transform=ax1.transAxes, fontsize=16,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.6', linewidth=0.5))
ax1.text(0.97, 0.78, 'LCZ Distribution', transform=ax1.transAxes, fontsize=16,
         color='red', fontweight='bold', verticalalignment='top', horizontalalignment='right')


# ── Right: LST temperature distribution ──────────────────────────
ax2.bar(bin_centers, binned_counts, width=bin_width * 0.85, color=bar_color, edgecolor='white', linewidth=0.3)

# PCHIP for LST too
pchip2 = PchipInterpolator(bin_centers, binned_counts)
x_smooth2 = np.linspace(bin_centers[0], bin_centers[-1], 500)
y_smooth2 = np.maximum(pchip2(x_smooth2), 0)
ax2.plot(x_smooth2, y_smooth2, color=curve_color, linewidth=1.0)

ax2.yaxis.set_major_locator(FixedLocator([0, 5e8, 1e9, 1.5e9, 2e9]))
ax2.yaxis.set_major_formatter(fmt)
ax2.set_ylim(0, 2.2e9)
ax2.set_xlabel('Temperature (\u00b0F)', fontsize=16)
ax2.set_ylabel('Pixels', fontsize=16)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_xlim(-55, 180)

total_pixels_lst = int(binned_counts.sum())
mean_t = np.average(bin_centers, weights=binned_counts)
variance = np.average((bin_centers - mean_t)**2, weights=binned_counts)
std_t = np.sqrt(variance)
txt2 = f'Total pixels: ~{total_pixels_lst / 1e9:.1f}B\nMean: {mean_t:.1f}\u00b0F  Std: {std_t:.1f}\u00b0F'
ax2.text(0.97, 0.97, txt2, transform=ax2.transAxes, fontsize=16,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.6', linewidth=0.5))
ax2.text(0.97, 0.78, 'Temperature Distribution', transform=ax2.transAxes, fontsize=16,
         color='red', fontweight='bold', verticalalignment='top', horizontalalignment='right')


plt.savefig(OUTPUT_DIR / 'combined_distribution.png', dpi=300, facecolor='white')
plt.close()
print('Saved combined_distribution.png')
