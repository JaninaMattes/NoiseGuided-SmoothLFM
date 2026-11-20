# Changelog

## Unreleased

### Added
- `train.py`: single Hydra entry point for both Stage-2 models (replaces `train_bvae.py` / `train_rf.py`).
- `Makefile`: `install`, `test`, `smoke`, `train-bvae`, `train-decoder`, `experiment`, `mlflow-ui`, `clean`.
- `ldm/tracking.py`: MLflow alongside W&B (plus CSV, TensorBoard fallback); images are logged to every backend,
  switchable with `log_images`. Run info and the resolved `config.yaml` are attached to each run.
- Configs: thesis β-VAE `model/skip-vit-bvae-b-2-t2i`, debug models `skip-vit-bvae-s-2-t2i` and `sit-s-2`,
  `data/dummy256_latents`, and `experiment/smoke/{bvae,decoder}` for checkpoint-free CPU runs.
- Tests: config integrity (every config composes, every `target:` imports) and tracking backends.

### Changed
- `source_timestep` / `target_timestep` are root config keys shared by trainer and HDF5 data configs.
- Thesis experiment configs (`imnet256/*`) now reference existing configs and the thesis hyper-parameters.
- `configs/flow/sit-b-4.yaml` renamed to `sit-xl-2.yaml` (it configures SiT-XL/2).
- README rewritten around the thesis, with IEEE references and a link to the defence slides.

### Fixed
- Restored modules the trainers import but that were missing on this branch: `ldm/ema.py`, `ldm/metrics/`,
  `ldm/models/nn/dist/`, `ldm/tools/fourier/`, and the β-VAE trainer (`ldm/trainer_bvae_ti2.py`).
- `ldm/frameworks/beta-vae` renamed to the importable `beta_vae`.
- Training crashed on CPU-only machines (missing `devices` key when computing the global batch size).
- Images were only sent to the first logger and assumed TensorBoard otherwise.

### Removed
- Duplicate `configs/rf_config_v01.yaml` (identical to `rf_config.yaml`); duplicate body of `ldm/utils/helpers.py`
  (now re-exports `ldm/helpers.py`).
