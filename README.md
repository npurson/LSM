<h2 align="center"> <a href="https://arxiv.org/abs/2410.18956"> Large Spatial Model: End-to-end Unposed Images to Semantic 3D

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2403.20309-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.18956)
[![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kairunwen/LSM)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://largespatialmodel.github.io/)

</h5>

<div align="center">
This repository is the official implementation of the Large Spatial Model.
LSM reconstructs explicit radiance fields from two unposed images in real-time, capturing geometry, appearance, and semantics.
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Feature and RGB Rendering](#feature-and-rgb-rendering)
  - [Feature Visualization](#feature-visualization)
  - [RGB Color Rendering](#rgb-color-rendering)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Updates](#updates)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


## Feature and RGB Rendering

### Feature Visualization
https://github.com/NVlabs/LSM/raw/main/assets/demo_videos/output_fmap_video.mp4

### RGB Color Rendering
https://github.com/NVlabs/LSM/raw/main/assets/demo_videos/output_images_video.mp4

## Get Started

### Installation
0. **Dowload repo:**
   ````
   git clone --recurse-submodules https://github.com/NVlabs/LSM.git
   ````
1. **Create and activate conda environment:**
   ````bash
   conda create -n lsm python=3.10
   conda activate lsm
   ````

2. **Install PyTorch and related packages:**
   ````bash
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
   ````

3. **Install other Python dependencies:**
   ````bash
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ````

4. **Install PointTransformerV3:**
   ````bash
   cd submodules/PointTransformerV3/Pointcept/libs/pointops
   python setup.py install
   cd ../../../../..
   ````

5. **Install 3D Gaussian Splatting modules:**
   ````bash
   pip install submodules/3d_gaussian_splatting/diff-gaussian-rasterization
   pip install submodules/3d_gaussian_splatting/simple-knn
   ````

6. **Install OpenAI CLIP:**
   ````bash
   pip install git+https://github.com/openai/CLIP.git
   ````

7. **Build croco model:**
   ````bash
   cd submodules/dust3r/croco/models/curope
   python setup.py build_ext --inplace
   cd ../../../../..
   ````

8. **Download pre-trained models:**

   The following three model weights need to be downloaded:

   ```bash
   # 1. Create directory for checkpoints
   mkdir -p checkpoints/pretrained_models

   # 2. DUSt3R model weights
   wget -P checkpoints/pretrained_models https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

   # 3. LSEG demo model weights
   gdown 1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7 -O checkpoints/pretrained_models/demo_e200.ckpt

   # 4. LSM final checkpoint
   gdown 1q57nbRJpPhrdf1m7XZTkBfUIskpgnbri -O checkpoints/pretrained_models/checkpoint-final.pth
   ```

### Usage
1. Data preparation
   - Prepare any two images of indoor scenes (preferably indoor images, as the model is trained on indoor scene datasets).
   - Place your images in a directory of your choice.

   Example directory structure:
   ````bash
   demo_images/
   └── indoor/
       ├── scene1/
       │   ├── image1.jpg
       │   └── image2.jpg
       └── scene2/
           ├── room1.png
           └── room2.png
   ````

2. Commands
   ````bash
   # Reconstruct 3D scene and generate video using two images
   bash scripts/infer.sh
   ````

   Optional parameters in `scripts/infer.sh` (default settings recommended):
   ```bash
   # Path to your input images
   --file_list "demo_images/indoor/scene2/image1.jpg" "demo_images/indoor/scene2/image2.jpg"

   # Output directory for Gaussian points and rendered video
   --output_path "outputs/indoor/scene2"

   # Image resolution for processing
   --resolution "256"
   ```

## Updates

**[2025-03-06]** Added ScanNet data preprocessing pipeline improvements. For detailed instructions, please refer to [data_process/data.md](data_process/data.md).

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [DUSt3R](https://github.com/naver/dust3r)
- [Language-Driven Semantic Segmentation (LSeg)](https://github.com/isl-org/lang-seg)
- [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3)
- [pixelSplat](https://github.com/dcharatan/pixelsplat)
- [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{fan2024largespatialmodelendtoend,
      title={Large Spatial Model: End-to-end Unposed Images to Semantic 3D},
      author={Zhiwen Fan and Jian Zhang and Wenyan Cong and Peihao Wang and Renjie Li and Kairun Wen and Shijie Zhou and Achuta Kadambi and Zhangyang Wang and Danfei Xu and Boris Ivanovic and Marco Pavone and Yue Wang},
      year={2024},
      eprint={2410.18956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.18956},
}
```
