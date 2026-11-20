# Experiment tracking with MLflow

MLflow is on by default. Every training run also writes `metrics.csv` and `config.yaml` to its run directory
(`logs_dir/<name>/<date>/<time>/`), and you can add W&B alongside MLflow with `use_wandb=True`.

## Track and browse runs locally

```bash
make train-bvae                     # runs are recorded in ./mlruns
make mlflow-ui                      # http://127.0.0.1:5000
```

Each run records:

| What | Where in MLflow |
|---|---|
| Scalars (`train/loss`, `val/loss`, `lr-AdamW`, …) | Metrics |
| Trainer hyper-parameters (lr, β-VAE/decoder configs, timesteps, …) | Parameters |
| The full resolved config | `config.yaml` artifact |
| Command, batch size, devices, seed, … | Tags `info.*` |
| Sample grids (if `log_images=True`) | Artifacts under `images/` |

Runs are grouped by `mlflow_experiment` (default `image-ldm`). The smoke tests log to their own experiment, `smoke`.

## Common settings

```bash
make train-bvae ARGS="log_images=False"                      # scalars only
make train-bvae ARGS="mlflow_experiment=beta-ablation"       # group runs
make train-bvae ARGS="use_mlflow=False"                      # CSV only (TensorBoard is the fallback)
```

## Shared tracking server (recommended on clusters)

Start a server once on a machine that every node can reach, then point runs at it. Runs then live on the server,
not in a local `./mlruns`:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination /storage/mlflow-artifacts

export MLFLOW_TRACKING_URI=http://<server>:5000              # read by configs/base/train.yaml
make train-bvae
```

To move finished local runs off your machine, see [run_archive.md](run_archive.md).
