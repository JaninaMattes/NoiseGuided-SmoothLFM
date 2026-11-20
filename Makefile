# One-command entry points for NoiseGuided-SmoothLFM. Run `make` to list them.
#
# Typical order:  install -> weights -> dataset -> train-bvae -> train-decoder
# Variables (all optional):
#   ARGS         extra Hydra overrides, e.g. ARGS="source_timestep=0.2 model.params.beta=5.0 use_wandb=True"
#   EXP          experiment preset under configs/experiment/, e.g. EXP=smoke/flow_decoder_guided
#   BVAE_CKPT    frozen β-VAE checkpoint for the decoder (default: checkpoints/beta_vae/last.ckpt)
#   IMAGENET_DIR ImageNet in class-folder layout (train/, val/) for `make dataset`
#   RUNS_REMOTE  archive destination for `make archive-runs` (see docs/run_archive.md)

PYTHON      ?= uv run python
EXP         ?=
ARGS        ?=
BVAE_CKPT   ?=
RUNS_REMOTE ?=
DUMMY_CKPT  := checkpoints/dummy/beta_vae_skip_vit_s2.ckpt
LINT_PATHS  := train.py build_dataset.py dataprocessing/build.py dataprocessing/forward_ode.py \
               dataprocessing/hdf5_writer.py dataprocessing/sources.py ldm/tracking.py ldm/legacy.py \
               ldm/testing.py ldm/models/wrapper/pretrained_beta_vae.py ldm/dataloader/dataloader/hdf5_dataloader.py \
               scripts tests

TRAIN := HYDRA_FULL_ERROR=1 $(PYTHON) train.py
EXP_ARG = $(if $(EXP),experiment=$(EXP))

.DEFAULT_GOAL := help
.PHONY: help install lint test test-all snapshots smoke weights dummy-checkpoint dataset dataset-sample \
        train-bvae train-decoder experiment mlflow-ui archive-runs clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-17s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv with the pinned dependencies from uv.lock
	uv sync --locked

lint: ## Static checks (ruff) of the maintained modules
	uv run ruff check $(LINT_PATHS)

test: ## Fast test suite: configs, logic regression, data pipeline, tracking (~1 min)
	$(PYTHON) -m pytest -m "not slow and not weights"

test-all: ## Full suite incl. end-to-end training runs (slow) and real-checkpoint checks (if available)
	$(PYTHON) -m pytest

snapshots: ## Accept intended changes of the experiment setup (rewrites tests/snapshots/)
	$(PYTHON) -m pytest tests/test_snapshots.py --update-snapshots

smoke: dummy-checkpoint ## End-to-end CPU runs of every stage on stand-in data (MLflow experiment "smoke")
	HYDRA_FULL_ERROR=1 $(PYTHON) build_dataset.py experiment=smoke/dataset
	$(TRAIN) experiment=smoke/beta_vae $(ARGS)
	$(TRAIN) --config-name train_flow_decoder experiment=smoke/flow_decoder $(ARGS)
	$(TRAIN) --config-name train_flow_decoder experiment=smoke/flow_decoder_guided $(ARGS)

weights: ## Download the public pre-trained weights (SD autoencoder, SiT-XL/2) into checkpoints/
	$(PYTHON) -m scripts.download_weights --dir checkpoints

dummy-checkpoint: ## Write a random β-VAE checkpoint ($(DUMMY_CKPT)) for decoder tests
	$(PYTHON) -m scripts.make_dummy_checkpoint --output $(DUMMY_CKPT)

dataset: ## Generate the guidance-free HDF5 dataset from $IMAGENET_DIR (resumable)
	HYDRA_FULL_ERROR=1 $(PYTHON) build_dataset.py $(EXP_ARG) $(if $(IMAGENET_DIR),paths.imagenet=$(IMAGENET_DIR)) $(ARGS)

dataset-sample: ## Small real dataset from the repository sample images (needs `make weights`)
	HYDRA_FULL_ERROR=1 $(PYTHON) build_dataset.py experiment=dataset/sample $(ARGS)

train-bvae: ## Stage 2a: train the β-VAE encoder (thesis setting by default)
	$(TRAIN) $(EXP_ARG) $(ARGS)

train-decoder: ## Stage 2b: train the self-guided decoder, e.g. BVAE_CKPT=$(DUMMY_CKPT) EXP=smoke/flow_decoder_guided
	$(TRAIN) --config-name train_flow_decoder $(EXP_ARG) $(if $(BVAE_CKPT),paths.bvae_ckpt=$(BVAE_CKPT)) $(ARGS)

experiment: ## Any root config: make experiment CONFIG=train_flow_decoder EXP=... ARGS=...
	$(TRAIN) --config-name $(or $(CONFIG),train_beta_vae) $(EXP_ARG) $(ARGS)

mlflow-ui: ## Browse tracked runs at http://127.0.0.1:5000
	$(PYTHON) -m mlflow ui --backend-store-uri ./mlruns

archive-runs: ## Bundle finished runs and move them to $RUNS_REMOTE (ARGS="--prune --transport rclone")
	@test -n "$(RUNS_REMOTE)" || { echo "Set RUNS_REMOTE (see docs/run_archive.md)"; exit 1; }
	$(PYTHON) -m scripts.archive_runs --logs logs_dir --dest $(RUNS_REMOTE) $(ARGS)

clean: ## Remove smoke-test outputs and caches (real runs and datasets are kept)
	rm -rf logs_dir/smoke data/processed/smoke.hdf5 .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
