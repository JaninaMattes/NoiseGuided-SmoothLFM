# 🚀 NoiseGuided-SmoothLFM

**NoiseGuided-SmoothLFM** is a framework for learning **noise-guided latent smoothness** in image generation.  
It builds on top of **Image Latent Diffusion Model** ([image-ldm](https://github.com/joh-schb/image-ldm)) and leverages the power of 


**Scalable Interpolant Transformers (SiT)** ([SiT](https://github.com/willisma/SiT)).

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
  - UMAP projections
  - Full image metric tracking (FID, precision-recall, smoothness)
- `configs/` — Example training & sampling configurations.
- `environment.yml` — Conda environment file.

---

## 🔧 Setup

Clone and enter the repository:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
