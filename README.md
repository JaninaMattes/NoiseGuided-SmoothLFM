# 🚀 NoiseGuided-SmoothLFM




- We utilise Scalable Interpolant Transformers (SiT), a family of generative models built on the backbone of Diffusion Transformers (DiT). 
- The pre-trained DiT-XL/2 encoder of this work is publicly available under https://github.com/willisma/SiT on the conditional ImageNet 256x256 benchmark using the exact same backbone, number of parameters, and GFLOPs.


This repository contains:
* A training script using PyTorch




🔧  Setup
First, download and set up the repo:

git clone ...
cd .. 



We provide an environment.yml file that can be used to create a Conda environment. If you only want to run pre-trained models locally on CPU, you can remove the cudatoolkit and pytorch-cuda requirements from the file.

conda env create -f environment.yml
conda activate ldm-env