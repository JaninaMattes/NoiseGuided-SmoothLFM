#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=vitb_x02x10x5b

# ---------------------- Info
echo "################################"
echo "DATE: $(date +"%Y-%m-%d %T")"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_PROCID: ${SLURM_PROCID}"
echo "USER: $USER"
echo "NODES: ${SLURM_NNODES}"
echo "################################"

# ---------------------- Env Setup
cd /export/home/ra93jiz/dev/Img-IDM/
source /export/home/ra93jiz/miniconda3/bin/activate
conda activate ldm-env-v2

# ---------------------- NCCL / CUDA settings
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s 
export NCCL_IB_RETRY_CNT=10
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# ---------------------- Define Timesteps
export SOURCE_TIMESTEP=0.20
export TARGET_TIMESTEP=1.00
export BETA_VALUE=5.0
export LR_RATE=1e-4
export DATASET='imagenet256_hdf5_v0'

# Define job name dynamically based on timesteps
export JOB_NAME="bvae_vits2_run_${SOURCE_TIMESTEP}x-${TARGET_TIMESTEP}x"

# ---------------------- Define args
export ARGS="experiment=imnet256/bvae-skipvit-b-2 \
  trainer_params.accumulate_grad_batches=2 \
  data.params.batch_size=128 \
  data.params.val_batch_size=128 \
  data=${DATASET} \
  data.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.target_timestep=$TARGET_TIMESTEP \
  data.params.train.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.train.params.target_timestep=$TARGET_TIMESTEP \
  data.params.validation.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.validation.params.target_timestep=$TARGET_TIMESTEP \
  trainer_module.params.source_timestep=$SOURCE_TIMESTEP \
  trainer_module.params.target_timestep=$TARGET_TIMESTEP \
  trainer_params.check_val_every_n_epoch=1 \
  trainer_params.limit_val_batches=1 \
  trainer_params.precision=bf16-mixed \
  trainer_module.params.lr=$LR_RATE \
  model=skip-vit-bvae-b-2-t2i \
  model.params.beta=$BETA_VALUE \
  name=imnet256/beta-vae-skipViT-b-2/${DATASET}/${SOURCE_TIMESTEP}x-${TARGET_TIMESTEP}x-${BETA_VALUE}b/ \
  devices=1 \
  checkpoint_params.every_n_train_steps=5000 \
  +resume_checkpoint=null \
  autoencoder=sd_ae \
  metrics=noise_metrics \
  use_wandb=True"

# ---------------------- Launch
srun --ntasks-per-node=1 \
    python train_bvae_ti2.py ${ARGS} "num_nodes=${SLURM_NNODES}" "slurm_id=${SLURM_JOB_ID}"

# ---------------------- End
echo "########## JOB FINISHED ##########"
date
