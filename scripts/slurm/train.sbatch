#!/bin/bash
# Generic SLURM job for both training stages. Submit from the repository root, e.g.
#
#   sbatch --job-name=bvae_02x10x_5b --export=ALL,ARGS="source_timestep=0.20 model.params.beta=5.0" scripts/slurm/train.sbatch
#   sbatch --job-name=decoder_05x10x --export=ALL,CONFIG=train_flow_decoder,ARGS="paths.bvae_ckpt=<ckpt>" scripts/slurm/train.sbatch
#
# CONFIG: train_beta_vae (default) | train_flow_decoder;  ARGS: any Hydra overrides.
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate                 # created by `make install`

export NCCL_IB_TIMEOUT=50 UCX_RC_TIMEOUT=4s NCCL_IB_RETRY_CNT=10 NCCL_ASYNC_ERROR_HANDLING=1 PYTHONFAULTHANDLER=1
echo "$(date '+%F %T') job=${SLURM_JOB_ID} nodes=${SLURM_NNODES} config=${CONFIG:-train_beta_vae} args=${ARGS:-}"

# shellcheck disable=SC2086  # ARGS is a list of Hydra overrides
srun --ntasks-per-node=1 python train.py --config-name "${CONFIG:-train_beta_vae}" ${ARGS:-} \
    num_nodes="${SLURM_NNODES}" slurm_id="${SLURM_JOB_ID}" tqdm_refresh_rate=100
