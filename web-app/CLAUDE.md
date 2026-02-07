# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Static GitHub Pages site for visualizing Earthformer deep learning land surface temperature (LST) predictions for San Antonio, TX. Inference scripts run locally to pre-render PNG map tiles and JSON temperature grids, then the HTML loads everything from relative paths. No servers needed.

## Commands

```bash
# Create conda environment
conda env create -f environment.yml
conda activate earthformer

# Run inference + generate static tiles/JSON (takes a while)
python _inference_city.py

# Preview locally
python -m http.server 3000
# Open http://localhost:3000
```

## Architecture

### Data Flow
1. `_inference_city.py` — `CityWideInference` class handles the full pipeline: STAC data fetching (Landsat + NASADEM), band calculation, model inference, and static asset generation:
   - `tiles/{month}/{z}/{x}/{y}.png` — pre-rendered RGBA PNG tiles at z14-17
   - `temperature/{month}.json` — per-month temperature grid for JS point queries
   - `months.json` — time slider metadata and temperature ranges
2. `model.py` — `LandsatLSTPredictor` (PyTorch Lightning) wraps Earthformer `CuboidTransformerModel` or `DMVSTNet_Landsat` (CNN+LSTM)
3. `index.html` — Interactive 3D map (ArcGIS JS API 4.31) with buildings, trees, LST overlay, time slider, and chat sidebar. All data loaded from relative paths.

### Key Files
- `downtown_westside.geojson` — clip polygon for downtown San Antonio area
- `scene_cache/` — cached monthly Landsat band arrays (gitignored)
- `city_inference_output/` — intermediate prediction rasters (gitignored)
- `.nojekyll` — tells GitHub Pages to serve tile paths correctly

### Key Constants
- **Bounding box**: `[-98.7459, 29.2534, -98.2467, 29.7534]` (San Antonio)
- **Tile size**: 128x128 pixels at 30m resolution
- **Input channels** (9): DEM, LST, Red, Green, Blue, NDVI, NDWI, NDBI, Albedo
- **Temperature normalization**: Fahrenheit range [-189, 211], normalized to [0, 1]
- **NODATA sentinel**: 0 in normalized space
- **Render tile size**: 256x256 PNG at zoom levels 14-17
- **Color ramp**: 25-bin teal→yellow→red

### External Dependencies
- **Microsoft Planetary Computer** — STAC API for Landsat C2 L2 and NASADEM data
- **Earthformer** — CuboidTransformer model architecture (imported from `earthformer` package)
- **model_baseline.ckpt** — Pre-trained model checkpoint (tracked via Git LFS)
- **San_Antonio/*.shp** — City boundary shapefile for spatial filtering
