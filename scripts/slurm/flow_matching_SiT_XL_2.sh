#!/bin/bash

#SBATCH --time=15:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --job-name=SiT-XL2_02x10x01b_cls_drop_run

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

# ---------------------- Dynamic Parameters
export SOURCE_TIMESTEP=0.20
export BETAVAE_TARGET_TIMESTEP=1.00     # Target for BetaVAE  (1.0: data manifold)
export BETAVAE_MODEL_TYPE="BetaVAE-B-2" # Model type for BetaVAE
export TARGET_TIMESTEP=1.00             # Target for FM is always 1.0 (data manifold)
export BETA_VALUE=0.1
# Note: cls_cond_w_dropout is used for the context conditioning with dropout
export LR_RATE=5e-5
export NUM_CLASS=1000 # Number of classes for the dataset

export TIMESTEP_TAG="${SOURCE_TIMESTEP}x-${BETAVAE_TARGET_TIMESTEP}x_${BETA_VALUE}b"

# ---------------------- Define args
export ARGS="experiment=imnet256/sit-xl_context_cond \
  model=sit-xl-2_context \
  data=imagenet256_hdf5_v0 \
  data.params.batch_size=128 \
  data.params.val_batch_size=128 \
  data.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.target_timestep=$TARGET_TIMESTEP \
  data.params.train.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.train.params.target_timestep=$TARGET_TIMESTEP \
  data.params.validation.params.source_timestep=$SOURCE_TIMESTEP \
  data.params.validation.params.target_timestep=$TARGET_TIMESTEP \
  trainer_module.params.source_timestep=$SOURCE_TIMESTEP \
  trainer_module.params.target_timestep=$TARGET_TIMESTEP \
  trainer_module.params.lr=$LR_RATE \
  trainer_module.params.num_classes=$NUM_CLASS \
  +resume_checkpoint=null \
  trainer_params.limit_val_batches=10 \
  trainer_params.check_val_every_n_epoch=1 \
  trainer_params.accumulate_grad_batches=2 \
  trainer_params.precision=bf16-mixed \
  name=imnet256/SiT-XL-2/context_cls_cond_w_dropout/${TIMESTEP_TAG}/${BETAVAE_MODEL_TYPE}/V0 \
  devices=1 \
  checkpoint_params.every_n_train_steps=5000 \
  autoencoder=sd_ae \
  ldm_autoencoder=v0/dataset_2/02x10x_0.1b_vit_b_2.yaml \
  metrics=img_metrics \
  use_wandb=True"

# ---------------------- Launch
srun --ntasks-per-node=1 \
    python train_rf.py $ARGS \
    num_nodes=${SLURM_NNODES} slurm_id=${SLURM_JOB_ID}

# ---------------------- End
echo "########## JOB FINISHED ##########"
date