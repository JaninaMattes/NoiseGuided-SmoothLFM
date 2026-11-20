# One-command entry points for NoiseGuided-SmoothLFM. Run `make` to list them.
#
#   make install                 # create .venv with the pinned dependencies (uv)
#   make smoke                   # end-to-end CPU check of both stages, no data/checkpoints needed
#   make train-bvae              # Stage 2a: β-VAE self-guidance encoder (thesis config)
#   make train-decoder BVAE_CKPT=logs_dir/.../last.ckpt   # Stage 2b: self-guided SiT-XL/2 decoder
#
# Variables (all optional):
#   EXP   experiment under configs/experiment/, e.g. EXP=imnet256/bvae-skipvit-b-2
#   ARGS  extra Hydra overrides, e.g. ARGS="source_timestep=0.2 model.params.beta=5.0 use_wandb=True"

PYTHON    ?= uv run python
BVAE_EXP  ?= imnet256/bvae-skipvit-b-2
DEC_EXP   ?= imnet256/sit-xl-2_context_cond
BVAE_CKPT ?=
EXP       ?=
ARGS      ?=

TRAIN := HYDRA_FULL_ERROR=1 $(PYTHON) train.py

.DEFAULT_GOAL := help
.PHONY: help install test smoke train-bvae train-decoder experiment mlflow-ui clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install the pinned dependencies from uv.lock
	uv sync --locked

test: ## Fast checks: configs compose, all targets import, tracking backends receive images
	$(PYTHON) -m pytest

smoke: ## End-to-end CPU run of both stages on dummy data (logs to MLflow experiment "smoke")
	$(TRAIN) experiment=smoke/bvae $(ARGS)
	$(TRAIN) --config-name rf_config experiment=smoke/decoder $(ARGS)

train-bvae: ## Stage 2a: train the β-VAE encoder        (EXP=..., ARGS=...)
	$(TRAIN) experiment=$(or $(EXP),$(BVAE_EXP)) $(ARGS)

train-decoder: ## Stage 2b: train the self-guided decoder (BVAE_CKPT=..., EXP=..., ARGS=...)
	$(TRAIN) --config-name rf_config experiment=$(or $(EXP),$(DEC_EXP)) \
		$(if $(BVAE_CKPT),ldm_autoencoder.params.ckpt_path=$(BVAE_CKPT)) $(ARGS)

experiment: ## Any config: make experiment CONFIG=rf_config EXP=... ARGS=...
	$(TRAIN) --config-name $(or $(CONFIG),bvae_config) $(if $(EXP),experiment=$(EXP)) $(ARGS)

mlflow-ui: ## Browse tracked runs at http://127.0.0.1:5000
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns

clean: ## Remove smoke-test runs and Python caches (real runs in logs_dir/ are kept)
	rm -rf logs_dir/smoke .pytest_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
