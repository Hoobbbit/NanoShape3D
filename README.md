<div align="center">

# 🔬 NanoShape3D

### Automated 3D Morphology Reconstruction of Nanoparticles from Single 2D Electron Microscopy Images

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ITMO University](https://img.shields.io/badge/ITMO-Center%20for%20AI%20in%20Chemistry-red)](https://itmo.ru)

*Pavel Lutskiy · Julia Razlivina — Center for AI in Chemistry, ITMO University*

</div>

---

## Overview

NanoShape3D is an end-to-end pipeline that reconstructs **3D mesh models of nanoparticles** from ordinary 2D SEM/TEM micrographs — no sample preparation, no destructive sectioning, no specialised tomography hardware required.

The pipeline takes a single electron microscopy image as input and produces:
- **Segmentation masks** for every individual particle (including touching and overlapping ones)
- **RGBA cutouts** of each detected particle
- **Clustered representatives** grouped by morphological similarity
- **Watertight 3D meshes** (.glb) scaled to true physical dimensions in nanometres

<div align="center">
  <img src="assets/pipeline.png" alt="NanoShape3D Pipeline" width="800"/>
  <br/>
  <em>Full pipeline: 2D SEM/TEM image → YOLO12 scale detection → CellposeSAM segmentation → CLIP clustering → Hunyuan3D-2 3D reconstruction</em>
</div>

---

## Key Results

| Metric | Value |
|--------|-------|
| Segmentation IoU (CellposeSAM fine-tuned) | **0.84** |
| Scale-bar detection precision (YOLO26) | **0.994** |
| Scale-bar detection recall (YOLO26) | **0.933** |
| OCR recognition IoU (PaddleOCR fine-tuned) | **0.72** |
| Throughput vs FIB-SEM | **~10× faster** |
| Throughput vs confocal microscopy | **~5× faster** |
| Destructive to sample | **No** |

---

## Pipeline Architecture

NanoShape3D is composed of four sequential modules:

### 1. 🔍 Scale Detection & Physical Calibration
A **YOLOv8** model detects the scale-bar bounding box in the micrograph. **PaddleOCR** reads the numerical value and unit (nm / µm). The two together compute the `nm/px` conversion factor passed to the 3D reconstruction stage.

Diversity of scale-bar styles handled by the pipeline:

<div align="center">
  <img src="assets/scalebar_samples.png" alt="Scale bar diversity" width="600"/>
</div>

### 2. 🧩 Instance Segmentation (CellposeSAM)
A domain-adapted **CellposeSAM** model (Cellpose topology + SAM zero-shot features) segments individual nanoparticles, including touching and heavily overlapping configurations where standard watershed methods fail. Fine-tuned on 550 manually annotated SEM/TEM micrographs via CVAT.

### 3. 🗂️ Morphology Clustering
**CLIP ViT-L/14** embeddings are extracted for each particle cutout. **K-Means** with silhouette-score selection finds the optimal number of clusters and identifies the representative particle closest to each cluster centroid. This eliminates redundant 3D inference calls for morphologically identical particles.

### 4. 🧊 Generative 3D Reconstruction (Hunyuan3D-2)
**Hunyuan3D-2** lifts each 2D cutout to a watertight triangular mesh via:
1. **ShapeVAE + DiT** → compact latent 3D features
2. **SDF prediction** → mesh extraction via Marching Cubes
3. **Scale integration** → mesh rescaled to true nanometre dimensions

---

## Repository Structure

```
nanoshape3d/
├── notebooks/
│   └── nanoshape3d.ipynb      # Main end-to-end pipeline notebook (Kaggle / Colab)
├── assets/
│   ├── pipeline.png           # Architecture diagram
│   └── scalebar_samples.png   # Scale-bar diversity examples
├── models/
│   └── README.md              # Model download instructions
├── test_images/
│   └── README.md              # Place your .jpg/.png images here
├── results/                   # Output meshes (.glb) saved here
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Quickstart

### Option A — Kaggle (recommended, free GPU)

1. Upload `notebooks/nanoshape3d.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Enable **GPU accelerator** (Settings → Accelerator → GPU T4 x2)
3. Run all cells — the first setup cell downloads models and test images automatically via `gdown`

### Option B — Local Installation

**Requirements:** Python 3.10+, CUDA 11.8+ GPU

```bash
# 1. Clone the repository
git clone https://github.com/your-org/nanoshape3d.git
cd nanoshape3d

# 2. Create environment
conda env create -f environment.yml
conda activate nanoshape3d

# 3. Install CellposeSAM
pip install git+https://www.github.com/mouseland/cellpose.git

# 4. Install Hunyuan3D-2
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2 && pip install -r requirements.txt && pip install -e .
cd ..

# 5. Download pre-trained models
python -c "
import gdown
gdown.download_folder(
    'https://drive.google.com/drive/folders/1KizYc5lkyL1itwhEmZ9SBDKix-O9tZ8Q?usp=sharing',
    output='models/', quiet=False
)
"

# 6. Place your SEM/TEM images in test_images/ and open the notebook
jupyter notebook notebooks/nanoshape3d.ipynb
```

---

## Usage

Once the notebook is running, the workflow is fully interactive:

| Widget | Description |
|--------|-------------|
| `mask_image_number` | Select which micrograph to segment (slider 0 → N-1) |
| `cutout_to_png` | Export a specific particle cutout to PNG |
| `image_to_3D` | Select which cluster representative to reconstruct in 3D |

Output `.glb` meshes are saved to `results/` and can be viewed in any GLTF-compatible viewer (e.g., [gltf.report](https://gltf.report), Blender, Windows 3D Viewer).

---

## Model Weights

| Model | Purpose | Download |
|-------|---------|---------|
| `cellpose_sam_aug_epoch_0045` | Nanoparticle instance segmentation | [Google Drive](https://drive.google.com/drive/folders/1KizYc5lkyL1itwhEmZ9SBDKix-O9tZ8Q?usp=sharing) |
| `tencent/Hunyuan3D-2` | Single-view 3D mesh generation | [HuggingFace Hub](https://huggingface.co/tencent/Hunyuan3D-2) |
| YOLO26 scale-bar detector | Scale-bar localisation | Included in Google Drive folder above |
| Fine-tuned PaddleOCR | Scale-bar text recognition | Included in Google Drive folder above |

---

## Datasets

| Sub-dataset | Size | Annotation | Use |
|-------------|------|------------|-----|
| Scale-bar detection | 1 000 images | Bounding boxes | YOLO training |
| OCR recognition | 1 000 crops | Text labels | PaddleOCR fine-tuning |
| Segmentation | 550 images | Polygon instance masks (CVAT) | CellposeSAM fine-tuning |
| Raw collected | ~10 500 images | — | Filtered down to above |

Raw images were collected from scientific literature by students of ITMO University and supplemented with the [Kaggle EM Particle Segmentation dataset](https://doi.org/10.1021/acs.jcim.0c01455).

---

## Citation

If you use NanoShape3D in your research, please cite:

```bibtex
@article{lutskiy2025nanoshape3d,
  title   = {NanoShape3D: Automated 3D Morphology Reconstruction System
             for Nanomaterials from Single 2D Electron Microscopy Images},
  author  = {Lutskiy, Pavel and Razlivina, Julia},
  year    = {2025},
  institution = {Center for AI in Chemistry, ITMO University}
}
```

### Dependencies

This project builds on the following key works:

- **CellposeSAM** — Pachitariu et al., *bioRxiv* 2025 · [doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
- **Hunyuan3D-2** — Xiao et al. 2024 · [GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- **Segment Anything (SAM)** — Kirillov et al., *ICCV* 2023
- **YOLOv8** — Jocher et al. 2023 · [Ultralytics](https://github.com/ultralytics/ultralytics)
- **PaddleOCR** — PaddlePaddle Authors 2020 · [GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- **CLIP** — Radford et al., *ICML* 2021

---

## Roadmap
- [ ] Expanded segmentation dataset (550 → 2 000+ images)
- [ ] Streamlit / Gradio GUI — no-code interface for non-programmers
- [ ] Integration with materials property databases for structure–property mapping

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Made at <strong>Center for AI in Chemistry, ITMO University</strong> · Saint-Petersburg, Russia
</div>
