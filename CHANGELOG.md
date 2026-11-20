# Changelog

## Unreleased

### Added
- `train.py`: single Hydra entry point for both Stage-2 models (replaces `train_bvae.py` / `train_rf.py`).
- `build_dataset.py` + `dataprocessing/{build,forward_ode,sources,hdf5_writer}.py`: generation of the guidance-free
  HDF5 dataset (forward-time ODE of the frozen SiT-XL/2, ω = 1). It can resume, repairs half-written batches, refuses
  to mix generation settings in one file, and records provenance.
- `Makefile`: `install`, `lint`, `test`, `test-all`, `snapshots`, `smoke`, `weights`, `dummy-checkpoint`, `dataset`,
  `dataset-sample`, `train-bvae`, `train-decoder`, `experiment`, `mlflow-ui`, `archive-runs`, `clean`.
- `ldm/tracking.py`: MLflow alongside W&B (plus CSV, with TensorBoard as fallback). Images go to every backend and can
  be switched off with `log_images`. Run info and the resolved `config.yaml` are attached to each run.
- `ldm/legacy.py`: maps old module names to new ones, so checkpoints saved under the old names still load.
- `ldm/testing.py`, `configs/first_stage/pooling_stub.yaml`, `configs/flow_encoder/dit_s2_random.yaml`: stand-ins
  that let the whole pipeline run on a CPU without weights.
- Scripts: `download_weights.py` (SD autoencoder, SiT-XL/2), `make_dummy_checkpoint.py` (random β-VAE checkpoint for
  `make train-decoder`), `archive_runs.py` (bundle finished runs and move them to remote storage),
  `slurm/train.sbatch` (one job script for both stages).
- Tests: config composition and target imports, frozen thesis configs (`tests/snapshots/`), forward ODE,
  HDF5 writer/reader, image sources, dataset build, legacy checkpoint names, strict β-VAE loading, run archiving,
  tracking, and end-to-end training (`slow`). Checks against the trained checkpoints are opt-in (`weights`,
  `NGSLFM_THESIS_CKPTS`).
- CI: `.github/workflows/ci-cd.yaml` runs `make lint` and `make test-all` on every push and pull request.
- Docs: `docs/architecture.md` (components and naming), `docs/mlflow.md`, `docs/run_archive.md`.

### Changed
- Configs are hierarchical: shared paths and training defaults are in `configs/base/`. The root configs
  `train_beta_vae.yaml` / `train_flow_decoder.yaml` (formerly `bvae_config` / `rf_config`) reproduce the thesis setting
  by default. `source_timestep` / `target_timestep` are root keys shared by trainer and data.
- Config groups renamed by role: `autoencoder/` → `first_stage/`, `flow/` → `flow_encoder/`,
  `ldm_autoencoder/` → `guidance_encoder/` (one config, checkpoint via `paths.bvae_ckpt` / `BVAE_CKPT`). Models follow
  `<role>_<backbone>_<size><patch>`, e.g. `skip-vit-bvae-b-2-t2i` → `beta_vae_skip_vit_b2`. Class subsets moved to
  `configs/subset/`.
- Modules renamed: `trainer_bvae_ti2` → `trainer_beta_vae`, `trainer_rf_vae` → `trainer_flow_decoder`,
  `*_t2i` → plain names (the suffix never meant text-to-image), `*_dctc` → `*_dct_conditioned`,
  `wrapper/beta_vae-t2i` → `wrapper/pretrained_beta_vae` (now loads weights strictly).
- README rewritten around the thesis, with IEEE references and a link to the defence slides.
- `ruff` pinned as a dev dependency.

### Fixed
- Restored modules the trainers import but that were missing on this branch: `ldm/ema.py`, `ldm/metrics/`,
  `ldm/models/nn/dist/`, `ldm/tools/fourier/`.
- `ldm/frameworks/beta-vae` renamed to the importable `beta_vae`.
- Training crashed on CPU-only machines (missing `devices` key when computing the global batch size).
- `encode_second_stage` raised `UnboundLocalError` when no flow encoder was configured.
- Images were only sent to the first logger and assumed TensorBoard otherwise.
- Package `__init__.py` files appended their own directory to `sys.path`, letting modules shadow top-level names.
- `scripts/profiling/profile_module.py` built a trainer class that no longer exists.

### Removed
- Superseded architecture revisions (`*_t1i`, older Skip-ViT/ViT copies), unused `ldm/models/smooth_diffusion/` and
  `ldm/models/stable_diffusion/`, and the old data pipelines (`dataprocessing/{pipeline,sampler,utils,webdataset}/`).
- `configs/config.yaml`, `configs/train/`, `configs/rf_config_v01.yaml` (duplicate), pixel-space experiments,
  per-checkpoint `configs/ldm_autoencoder/*`, and the per-experiment SLURM scripts.
- Duplicate body of `ldm/utils/helpers.py` (now re-exports `ldm/helpers.py`).
