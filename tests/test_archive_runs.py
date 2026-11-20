"""Finished runs are bundled with a checksum manifest, copied to the destination and optionally pruned."""
import json
import os
import shutil
import tarfile
import time

import pytest

from scripts.archive_runs import STUB, archive_runs, sha256

pytestmark = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")


def make_run(logs, name: str, age_minutes: float):
    run = logs / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text("lr: 1e-4\n")
    (run / "metrics.csv").write_text("step,train/loss\n0,1.0\n")
    (run / "checkpoints" / "last.ckpt").write_bytes(b"weights")
    stamp = time.time() - 60 * age_minutes
    for path in run.rglob("*"):
        os.utime(path, (stamp, stamp))
    return run


def test_finished_run_is_archived_and_pruned(tmp_path):
    logs, dest = tmp_path / "logs", tmp_path / "remote"
    run = make_run(logs, "beta_vae/2025-07-01/T101500", age_minutes=60)
    make_run(logs, "beta_vae/2025-07-01/T235900", age_minutes=1)          # still running: skipped

    assert archive_runs(logs, str(dest), prune_local=True) == [run]

    archive = dest / "beta_vae__2025-07-01__T101500.tar.gz"
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        manifest = json.load(tar.extractfile("beta_vae__2025-07-01__T101500/MANIFEST.json"))
    assert {n.split("/", 1)[1] for n in names} == {"config.yaml", "metrics.csv", "MANIFEST.json"}  # no checkpoints
    assert set(manifest["files"]) == {"config.yaml", "metrics.csv"}

    stub = json.loads((run / STUB).read_text())
    assert stub["sha256"] == sha256(archive) and stub["pruned"]
    remaining = sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file())
    assert remaining == [STUB, "checkpoints/last.ckpt"]                    # only archived files are pruned

    assert archive_runs(logs, str(dest)) == []                             # archived runs are not repeated


def test_dry_run_changes_nothing(tmp_path):
    logs, dest = tmp_path / "logs", tmp_path / "remote"
    run = make_run(logs, "run", age_minutes=60)

    archive_runs(logs, str(dest), prune_local=True, dry_run=True)

    assert not any(dest.iterdir()) and (run / "metrics.csv").exists() and not (run / STUB).exists()
