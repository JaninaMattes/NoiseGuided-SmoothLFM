<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<h1 align="center"> Noise Guided Smooth LFM</h1>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JaninaMattes/NoiseGuided-SmoothLFM.git">
    <img src="assets/logo/img2.png" alt="Logo" width="100" height="100">
  </a>

<h3 align="center">Exploring Self-Supervised Representation Learning for Interpretability and Control in Flow and Diffusion-based Generative Models</h3>

  <p align="center">
    The purpose of this work is to explore how established self-supervised representation learning methods can be utilised for interpreting and guiding the behaviour of state-of-the-art Flow and Diffusion-based generative models.
    <br />
    <a href="https://github.com/JaninaMattes/NoiseGuided-SmoothLFM"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/JaninaMattes/NoiseGuided-SmoothLFM">View Demo</a>
    ·
    <a href="https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/issues">Report Bug</a>
    ·
    <a href="https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/issues">Request Feature</a>
  </p>
</div>



<!-- <p align="center">
  <img src="assets/readme/pair_010.gif" alt="Interpolation 1" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_04.gif" alt="Interpolation 2" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_06.gif" alt="Interpolation 3" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_05.gif" alt="Interpolation 4" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
</p>

<p align="center">
  <img src="assets/readme/pair_09.gif" alt="Interpolation 5" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_10.gif" alt="Interpolation 6" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_08.gif" alt="Interpolation 7" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
  <img src="assets/readme/pair_07.gif" alt="Interpolation 8" width="22%" style="margin:10px; background-color:#f0f0f0; border-radius:10px; padding:5px;">
</p> -->

This repo contains PyTorch model definitions, pre-trained weights and training/sampling code for experiments over ImageNet 256 × 256.

<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">

<table>
  <tr>
    <td colspan="2" align="center">
      <img src="assets/readme/pair_08.gif" width="100%">
    </td>
    <td colspan="2" align="center">
      <img src="assets/readme/pair_04.gif" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center"><img src="assets/readme/pair_010.gif" width="100%"></td>
    <td align="center"><img src="assets/readme/pair_23.gif" width="100%"></td>
    <td align="center"><img src="assets/readme/pair_06.gif" width="100%"></td>
    <td align="center"><img src="assets/readme/pair_35.gif" width="100%"></td>
  </tr>
</table>

</div>

## Background: Diffusion Models & Gaussian Flow Matching

Flow Matching and Diffusion Models are two dominant frameworks for generative modeling in vision. This work treats them as interchangeable, a choice that is mathematically justified: **ODE-based Diffusion Models and Gaussian Flow Matching are equivalent** when the source distribution is Gaussian. Different parameterizations yield different noise schedules and loss weightings, but they define the same generative model.

---

### Diffusion Models

A diffusion process gradually corrupts a data point $\mathbf{x}$ (e.g. an image) by progressively mixing it with Gaussian noise:
```math
\mathbf{z}_t = \alpha_t\,\mathbf{x} + \sigma_t\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
```

where $\alpha_t$ and $\sigma_t$ define the **noise schedule**, whereby this work uses the **variance-preserving** schedule ($\alpha_t^2 + \sigma_t^2 = 1$), ensuring the process transitions smoothly from clean data at $t=0$ to pure noise at $t=1$.

<details>
<summary><b>Reverse Process & DDIM Sampler</b></summary>
<br>

To generate new samples, we reverse the forward process:

1. **Initialize** $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. **Denoise**: predict the clean sample with a neural network (denoiser): $\hat{\mathbf{x}} = \hat{\mathbf{x}}(\mathbf{z}_t;\, t)$
3. **Project back** to a lower noise level $s < t$:
```math
\mathbf{z}_{s} = \alpha_{s}\,\hat{\mathbf{x}} + \sigma_{s}\,\hat{\boldsymbol{\epsilon}}, \qquad \hat{\boldsymbol{\epsilon}} = \frac{\mathbf{z}_t - \alpha_t\,\hat{\mathbf{x}}}{\sigma_t}
```

4. **Repeat** steps 2–3 from $t=1$ toward $t=0$ until $\hat{\mathbf{x}}$ is recovered.

> [!NOTE]
> This is the **DDIM sampler**. All stochasticity is concentrated in the initial sample $\mathbf{z}_1$, the entire reverse process is deterministic.

</details>

---

### Flow Matching

The forward process is a **linear interpolation** between data $\mathbf{x}$ and noise $\boldsymbol{\epsilon}$:
```math
\mathbf{z}_t = (1 - t)\,\mathbf{x} + t\,\boldsymbol{\epsilon}, \qquad t \in [0, 1]
```

This recovers the diffusion forward process under the schedule $\alpha_t = 1-t,\; \sigma_t = t$. The evolution between timesteps $s < t$ is then **linear**:
```math
\mathbf{z}_t = \mathbf{z}_s + \mathbf{u}\,(t - s)
```

where $\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{x}$ is the **velocity field**. Rather than predicting noise (as in DDPM), **our model learns to predict this velocity directly** — the straight-line path from data to noise.

> [!NOTE]
> Straight-line trajectories between source (e.g. _Gaussian noise_) and target distribution (e.g. _real ImageNet 256 x 256 image data_) reduce curvature, requiring fewer NFE (Number of Function Evaluations) at inference compared to curved diffusion paths which makes generation cheaper and faster.

---

### Probability Flow ODE

Song et al. show that any stochastic diffusion Stochastic Differential Equation (SDE) can be reformulated as a deterministic **probability flow ODE**, while preserving identical marginal densities $p_t(\mathbf{x})$ at every timestep. This means noise injection is not required during generation.

<p align="center">
  <img src="assets/diagrams/diagram_ode_sde_sampling.png" alt="SDE vs ODE sampling diagram" width="50%">
</p>

<details>
<summary><b>SDE → ODE Equivalence</b></summary>
<br>

*(Add your derivation or Song et al. formulation here.)*

</details>

New samples are generated from $\mathbf{z}_1 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ by discretising the reverse-time ODE with a numerical solver — this work uses the **first-order Euler method**.

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Architecture

The framework follows the standard Latent Diffusion Model architecture introduced by Stable Diffusion, adopting a two-stage process that operates entirely in a learned lower-dimensional latent space. The β-Variational Autoencoder follows a Vision Transformer (ViT)-based architecture, whereas the diffusion backbone is based on Scalable Interpolant Transformers (SiT).

### Stage 1: Semantic Compression

<p align="center">
  <img src="assets/diagrams/framework_stage1.png" alt="Framework Architecture Diagram" width="80%" style="border-radius:10px; background-color:#2e2e2e; padding:10px;">
</p>
An open-source, pre-trained, Stable Diffusion CNN-VAE encoder and decoder are utilised for semantic compression and decompression over ImageNet 256 × 256. Together, they are defining the fixed latent space in which the Conditional Flow Matching module is then trained. Operating in a slightly compressed latent space reduces not only computational complexity, but also benefits from a lower-variance, more regularised Gaussian latent space.

#### Results after CNN-VAE Compression
<p align="center">
  <img src="assets/diagrams/latent_noise_codes0.png" alt="Framework Architecture Diagram" width="80%" style="border-radius:10px; background-color:#2e2e2e; padding:10px;">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Stage 2: Latent Conditional Flow Matching

<p align="center">
  <img src="assets/diagrams/framework_stage2.png" alt="Framework Architecture Diagram" width="80%" style="border-radius:10px; background-color:#2e2e2e; padding:10px;">
</p>

Both the Flow Matching module and the β-Variational Autoencoder operate entirely within the fixed autoencoder latent space, thereby shaping the prior distribution. Together the CFM module allows for abstract feature information extraction and detailed object appearance recovery. 

#### Building "Guidance-Free" Noise Spaces
<p align="center">
  <img src="assets/diagrams/forward_ode_noise.png" alt="Framework Architecture Diagram" width="50%" style="border-radius:10px; background-color:#2e2e2e; padding:10px;">
</p>

The idea of this work is founded on the idea of deterministically learned noise spaces, referred to as “Guidance-Free”, since we are able to reconstruct high-quality images from such noise samples without requiring additional classifier-free guidance.

#### Results after Forward Diffusion

The ODE-based forward diffusion process generates linearly dependent 
<table style="height: 50%; width: 100%; table-layout: fixed;">

  <tr>
    <td><img src="assets/diagrams/latent_noise_codes1.png"  width="100%"></td>
  </tr>
  <tr>
    <td><img src="assets/diagrams/latent_noise_codes2.png"  width="100%"></td>
  </tr>
    <tr>
    <td><img src="assets/diagrams/latent_noise_codes3.png"  width="100%"></td>
  </tr>
</table>

<p align="center">
  <img src="assets/diagrams/latent_representations.png" alt="Framework Architecture Diagram" width="80%" style="border-radius:10px; background-color:#2e2e2e; padding:10px;">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>




### Built With
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-%23ee4c2c?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green?logo=nvidia)
![License](https://img.shields.io/badge/license-MIT-lightgrey)





<!-- ## ✨ Highlights

- 🌀 **Noise-guided latent smoothness** for continuous morphing and perceptually stable interpolations.
- 🚀 Built on **Scalable Interpolant Transformers (SiT)** and latent diffusion.
- 🔬 Extensive quantitative evaluations: PCA, UMAP, linear probes, ISTD, LDPL metrics.
- 🎥 Supports advanced metrics: FID, Inception Score, LPIPS, SSIM, PSNR, and custom smoothness metrics.
- 💥 Enables **creative editing**, seamless transitions, and robust self-guidance. -->


## 🌊 Smooth Interpolations

Large-scale generative models such as Diffusion Models and the recent Flow Matching (FM) paradigm have demonstrated remarkable synthesis capabilities (Dhariwal & Nichol, 2021; Lipman et al., 2023; Albergo et al., 2023). However, their capacity for robust and structured representation learning remains largely underexplored. Continuous-time architectures, including Scalable Interpolant Transformers (SiT) (Ma et al., 2024), have yet to be rigorously evaluated regarding their ability to learn compact, semantically meaningful latent spaces.

By design, Diffusion and Flow-based models lack explicit architectural constraints or dedicated modules for enforcing smooth, disentangled feature extraction. While this choice preserves sample fidelity and diversity, it inherently limits interpretability and fine-grained control (Fuest et al., 2024).

In contrast to supervised or heavily engineered methods, our approach introduces a fully self-supervised, representation-learning-based guidance mechanism. By integrating a tunable β-VAE encoder, we extract compact, smooth latent codes directly from pretrained generative backbones. This enables semantically coherent interpolations without external annotations or handcrafted constraints, providing a scalable and interpretable solution.

Evaluating interpolation behavior — for example, via linear interpolations or latent space walks — offers an intuitive and interpretable means of assessing representation quality. While prior works have explored smoother latent traversals and morphing capabilities (Guo et al., 2024; Zhang et al., 2024), these typically rely on explicit supervision, complex augmentation pipelines, or auxiliary conditioning networks, which introduce additional complexity and reduce scalability.

Our framework is lightweight, architecture-agnostic, and directly applicable to a wide range of Diffusion and Flow-based backbones without retraining.

> 🎯 **Our method enables smooth, continuous transitions between images while preserving fine-grained semantic details and global structure.** This facilitates creative interpolations, intuitive attribute editing, and robust exploratory latent space walks — all while maintaining high sample quality and diversity.

---

## 💡 Motivation

This framework tackles core trade-offs in generative modeling- **sample quality, diversity, and inference speed** —while introducing a principled path to greater **interpretability and controllability** without relying on annotated datasets. It achieves this by integrating an auxiliary Bayesian β-VAE encoder and leveraging deterministic continuous flows instead of stochastic noise schedules, yielding smoother, more structured, and more controllable outputs.

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/JaninaMattes/NoiseGuided-SmoothLFM.git
cd NoiseGuided-SmoothLFM

# Create environment
conda create -n ldm-env python=3.12
conda activate ldm-env

# Install core packages
conda install pytorch=2.5.1 torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install lightning=2.5.0 -c conda-forge
conda install -c conda-forge pillow matplotlib einops timm h5py pandas webdataset tensorboard wandb

# Additional packages
pip install hydra-core --upgrade
pip install torch-fidelity torchdiffeq open_clip_torch notebook lpips pytorch-fid moviepy umap-learn
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/joh-schb/jutils.git#egg=jutils

# Run training
python train.py --config configs/your_config.yaml
```
---

## Rsource
@inproceedings{Song2020score,
  author    = {Yang Song and Jascha Sohl-Dickstein and Diederik P. Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  title     = {Score-Based Generative Modeling through Stochastic Differential Equations},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020}
}

@book{Murphy2023probabilistic,
  author    = {Kevin P. Murphy},
  title     = {Probabilistic Machine Learning: Advanced Topics},
  publisher = {MIT Press},
  year      = {2023}
}


[0] Dhariwal & Nichol (2021), "Diffusion Models Beat GANs on Image Synthesis."

[1] Ma et al. (2024), "SiT: Stochastic interpolant transport for generative modeling."

[2] Guo et al. (2024), "Smooth Diffusion: Crafting Smooth Latent Spaces in Diffusion Models."

[3] Zhang et al. (2024), "DiffMorpher: Unleashing the Capability of Diffusion Models for Image Morphing."

[4] Fuest et al. (2024), "Diffusion Models and Representation Learning: A Survey."

[5] Lipman et al. (2023), "Flow Matching for Generative Modeling."

[7] Albergo et al. (2023), "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions."


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- https://github.com/JaninaMattes/NoiseGuided-SmoothLFM -->
[contributors-shield]: https://img.shields.io/github/contributors/JaninaMattes/NoiseGuided-SmoothLFM.svg?style=for-the-badge
[contributors-url]: https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JaninaMattes/NoiseGuided-SmoothLFM.svg?style=for-the-badge
[forks-url]: https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/network/members
[stars-shield]: https://img.shields.io/github/stars/JaninaMattes/NoiseGuided-SmoothLFM.svg?style=for-the-badge
[stars-url]: https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/stargazers
[issues-shield]: https://img.shields.io/github/issues/JaninaMattes/NoiseGuided-SmoothLFM.svg?style=for-the-badge
[issues-url]: https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/issues
[license-shield]: https://img.shields.io/github/license/JaninaMattes/NoiseGuided-SmoothLFM.svg?style=for-the-badge
[license-url]: https://github.com/JaninaMattes/NoiseGuided-SmoothLFM/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 