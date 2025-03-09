# Data Preparation Guide for ScanNet and ScanNet++

## Table of Contents
- [Data Preparation Guide for ScanNet and ScanNet++](#data-preparation-guide-for-scannet-and-scannet)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [ScanNet Data Preparation](#scannet-data-preparation)
    - [1. Download ScanNet Data](#1-download-scannet-data)
    - [2. Extract .sens Files](#2-extract-sens-files)
    - [3. Data Processing](#3-data-processing)
    - [4. Data Structure](#4-data-structure)
  - [ScanNet++ Data Preparation](#scannet-data-preparation-1)
    - [1. Download ScanNet++ Data](#1-download-scannet-data-1)
    - [2. Data Processing](#2-data-processing)
    - [3. Data Structure](#3-data-structure)

## Overview
This document provides instructions for preparing ScanNet and ScanNet++ datasets for training and evaluation.

## Prerequisites
- Python 3.7+
- Sufficient storage space (>1TB recommended)
- Required Python packages:
  ```bash
  pip install -r data_process/requirements.txt
  ```

## ScanNet Data Preparation

### 1. Download ScanNet Data
1. Visit the [official ScanNet repository](https://github.com/ScanNet/ScanNet)
2. Fill out the [Terms of Use agreement](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)
3. Send the signed agreement to scannet@googlegroups.com
4. You will receive download instructions and access credentials via email

Example download command (after receiving credentials):
```bash
# Download ScanNet v2 data
python download-scannet.py -o ./data/scannet --type .sens
```

### 2. Extract .sens Files
We provide a parallel processing script to efficiently extract data from .sens files:

```bash
# Run the parallel export script
python data_process/scannet/export_data/export.py \
    --input_dir ./data/scannet \
    --output_dir ./data/scannet_extracted \
    --num_workers 8
```

The export script provides the following features:
- Parallel processing using multiple CPU cores
- Automatic skipping of already processed scenes
- Progress tracking with tqdm

Arguments:
- `--input_dir`: Directory containing the ScanNet dataset
- `--output_dir`: Directory to save extracted data
- `--num_workers`: Number of parallel workers (default: half of CPU cores)

Output structure for each scene:
```
data/scannet_extracted/
├── scene0000_00/
│ ├── color/ # RGB images in jpg format
│ │ ├── 0.jpg
│ │ ├── 1.jpg
│ │ └── ...
│ ├── depth/ # Depth images in png format (16-bit, depth shift 1000)
│ │ ├── 0.png
│ │ ├── 1.png
│ │ └── ...
│ ├── pose/ # Camera poses (4x4 matrices)
│ │ ├── 0.txt
│ │ ├── 1.txt
│ │ └── ...
│ └── intrinsic/ # Camera parameters
│ ├── intrinsic_color.txt # Color camera intrinsics
│ ├── intrinsic_depth.txt # Depth camera intrinsics
│ ├── extrinsic_color.txt # Color camera extrinsics
│ └── extrinsic_depth.txt # Depth camera extrinsics
├── scene0000_01/
│ ├── color/
│ ├── depth/
│ ├── pose/
│ └── intrinsic/
└── ...
```

### 3. Data Processing
```bash
# Process raw data
python -m data_process.scannet.scannet_processor \
    --root_dir data/scannet_extracted \
    --save_dir data/scannet_processed \
    --device cuda \
    --num_workers 8
```

Arguments:
- `--root_dir`: Path to the extracted ScanNet data directory (default: "data/scannet_extracted")
  - Should contain the extracted .sens files organized by scene
  - Each scene should have color/, depth/, pose/, and intrinsic/ subdirectories

- `--save_dir`: Path where processed data will be saved (default: "data/scannet_processed")
  - Will create if directory doesn't exist
  - Processed data will be organized by scene with standardized format

- `--device`: Computing device to use (default: "cuda")
  - "cuda": Use GPU acceleration (recommended)
  - "cpu": Use CPU only (slower)

- `--num_workers`: Number of parallel processing workers (default: 8)
  - Higher values may speed up processing but use more memory
  - Recommended: Set to number of CPU cores or less

Note: Ensure sufficient disk space in save_dir (>500GB recommended for full dataset)

### 4. Data Structure
After processing, the ScanNet data will be organized in the 
following structure:
```
data/scannet_processed/
├── scene0000_00/
│   ├── color/         # Directory containing RGB images
│   │   ├── 000000.png
│   │   └── ...        # Additional RGB images
│   ├── depth/         # Directory containing Depth maps
│   │   ├── 000000.png
│   │   └── ...        # Additional Depth maps
│   └── pose/          # Directory containing Camera poses
│       ├── 000000.npz
│       └── ...        # Additional camera pose files
├── scene0000_01/
└── ...                 # Additional scenes
```

## ScanNet++ Data Preparation

### 1. Download ScanNet++ Data
1. Visit the [official ScanNet++ repository](https://github.com/scannetpp/scannetpp)
2. Fill out the Terms of Use agreement
3. You will receive a download script and token (valid for 14 days)

To download the dataset:
1. Navigate to the download script directory
2. Edit `download_scannetpp.yml` configuration file:
   - Set your token
   - Configure download directory
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the download script:
```bash
python download_scannetpp.py download_scannetpp.yml
```

### 2. Data Processing
For DSLR format data processing, follow these steps:

1. Create and setup rendering environment:
```bash
# Create new conda environment
conda create -n renderpy python=3.9
conda activate renderpy

# Install Python dependencies
pip install imageio numpy tqdm opencv-python pyyaml munch

# Clone and build renderpy
cd data_process/scannetpp
git clone --recursive https://github.com/liu115/renderpy
cd renderpy

# Install system dependencies
sudo apt-get install build-essential cmake git
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev libxrandr-dev libxext-dev
sudo apt-get install libopencv-dev
sudo apt-get install libgflags-dev libboost-all-dev

# Install renderpy
pip install . && cd ..
```

2. Configure rendering:
- Edit `data_process/scannetpp/common/configs/render.yml`
- Update `data_root` and `output_dir` paths

3. Run rendering process:
```bash
python -m common.render common/configs/render.yml
```


### 3. Data Structure
The processed ScanNet++ data will be organized as follows:
```
data/scannetpp_render/
└── {scene_id}/ # e.g., fb564c935d
    └── {device}/ # device can be 'dslr' or 'iphone'
        ├── camera/ # Camera parameter files
        │   ├── {frame_id}.npz # Contains intrinsic and extrinsic matrices
        │   └── ...
        ├── render_depth/ # Rendered depth maps
        │   ├── {frame_id}.png # 16-bit depth maps (depth * 1000)
        │   └── ...
        ├── rgb_resized_undistorted/ # Processed RGB images
        │   ├── {frame_id}.JPG # Undistorted and resized color images
        │   └── ...
        └── mask_resized_undistorted/ # Processed mask images
            ├── {frame_id}.png # Binary masks (0 or 255)
            └── ...
```

Each directory contains:
- `camera/`: Camera parameter files in .npz format, containing:
  - `intrinsic`: 3x3 camera intrinsic matrix
  - `extrinsic`: 4x4 camera-to-world transformation matrix
- `render_depth/`: Rendered depth maps stored as 16-bit PNG files (depth values * 1000)
- `rgb_resized_undistorted/`: Undistorted and resized RGB images
- `mask_resized_undistorted/`: Undistorted and resized binary mask images (255 for valid pixels, 0 for invalid)