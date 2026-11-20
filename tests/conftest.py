"""Shared fixtures: environment-independent config composition and a small generated dataset.

Markers (registered in pyproject.toml):
    slow     end-to-end training runs (``make test-all``)
    weights  checks against the trained thesis checkpoints in $NGSLFM_THESIS_CKPTS (skipped if unset)
"""
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
CONFIG_ENV_VARS = ("NGSLFM_CHECKPOINTS", "NGSLFM_DATA", "IMAGENET_DIR", "BVAE_CKPT", "MLFLOW_TRACKING_URI")


def pytest_addoption(parser):
    parser.addoption("--update-snapshots", action="store_true",
                     help="rewrite tests/snapshots/ with the current resolved configs (make snapshots)")


def compose_config(config_name: str, overrides: list[str] = ()) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch):
    """Configs read env variables; tests must see the repository defaults, not the caller's shell."""
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USER", "test")


@pytest.fixture(scope="session")
def smoke_dataset(tmp_path_factory) -> Path:
    """The smoke HDF5 dataset (2 classes, 4 train / 2 validation samples) built by the real pipeline."""
    from dataprocessing.build import build_dataset

    output = tmp_path_factory.mktemp("data") / "smoke.hdf5"
    build_dataset(compose_config("build_dataset", [
        "experiment=smoke/dataset", f"paths.data={REPO / 'data'}", f"output={output}",
    ]))
    return output
