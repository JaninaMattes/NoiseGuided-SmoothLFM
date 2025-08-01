# 🚀 NoiseGuided-SmoothLFM

![Smooth Interpolation Example](assets/readme/pair_04.gif)

**NoiseGuided-SmoothLFM** is a framework for learning **noise-guided latent smoothness** in image generation, allowing for improved, self-contained downstream controllability without the requirement for additional annotated data or class-based guidance.  
It builds on top of **Image Latent Diffusion Model** ([image-ldm](https://github.com/joh-schb/image-ldm)) and leverages the power of **Scalable Interpolant Transformers (SiT)** ([SiT](https://github.com/willisma/SiT)).

---

## 🌊 Smooth Interpolations

Our model learns *smooth transitions* in an auxiliary diffusion-based latent space, enabling natural morphing between images, as showcased above.  
This allows for creative interpolations and continuous edits that remain perceptually consistent.

---

## 💡 Highlights

- ✅ Uses **pre-trained SiT (Scalable Interpolant Transformers)** as the backbone, which are built on Diffusion Transformers (DiT).
- 🧊 Employs **smoothness constraints** in the latent space for better editing, interpolation, and stability.
- 🔬 Includes advanced quantitative evaluations: PCA projections, linear probes, UMAP visualizations, and smoothness metrics (ISTD, LDPL).
- 📈 Integrated support for FID, Inception Score, LPIPS, SSIM, PSNR, and custom smoothness metrics.
- 💥 Supports advanced ODE-based sampling via `torchdiffeq`.

---

## 📦 Repository Contents

- `train.py` — Main training entry point for SiT-based models, including smoothness losses.
- `evaluation/` — Scripts for:
  - PCA and linear probe evaluations
  - UMAP projections for non-linear evaluation of global structures
  - Full image metric tracking in latent and pixel domain (FID, precision-recall, smoothness)
- `configs/` — Example training & sampling configurations.
- `environment.yml` — Conda environment file.

---

## 🔧 Setup

Clone and enter the repository:

```bash
git clone git@github.com:JaninaMattes/NoiseGuided-SmoothLFM.git
cd NoiseGuided-SmoothLFM
