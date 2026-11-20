"""End-to-end training of every stage on the smoke dataset (CPU, a few steps each).

Runs ``train.py`` exactly as ``make smoke`` does, but writes everything below a temporary
directory. Marked ``slow``: run with ``make test-all``.
"""
import csv
import math
import os
import subprocess
import sys

import pytest

from conftest import REPO
from scripts.make_dummy_checkpoint import make_dummy_checkpoint

pytestmark = pytest.mark.slow

RUNS = {                                    # experiment -> root config
    "smoke/beta_vae": "train_beta_vae",
    "smoke/flow_decoder": "train_flow_decoder",
    "smoke/flow_decoder_guided": "train_flow_decoder",
}


@pytest.mark.parametrize("experiment", RUNS)
def test_training_run_logs_metrics_and_checkpoint(experiment, smoke_dataset, tmp_path):
    bvae_ckpt = make_dummy_checkpoint(tmp_path / "beta_vae.ckpt")
    command = [
        sys.executable, "train.py", "--config-name", RUNS[experiment], f"experiment={experiment}",
        f"dataset_file={smoke_dataset}", f"paths.logs={tmp_path / 'logs'}", f"paths.bvae_ckpt={bvae_ckpt}",
        f"mlflow_tracking_uri=file:{tmp_path / 'mlruns'}",
    ]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env={**os.environ, "HYDRA_FULL_ERROR": "1"})
    assert result.returncode == 0, result.stderr[-3000:]

    (run_dir,) = {p.parent for p in (tmp_path / "logs").rglob("config.yaml")}
    with open(run_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    losses = [float(r["train/loss"]) for r in rows if r.get("train/loss")]
    assert losses and all(map(math.isfinite, losses)), "no finite training loss logged"
    assert any(r.get("val/loss") for r in rows), "validation did not run"
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert list((tmp_path / "mlruns").rglob("meta.yaml")), "no MLflow run recorded"
