"""Every experiment config must compose and every ``target:`` in it must be importable.

Nothing is instantiated, so no data or checkpoints are needed. This guards against configs
pointing at modules or config groups that do not exist in the repository.
"""
import importlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

RUNS = [
    ("bvae_config", []),
    ("bvae_config", ["experiment=smoke/bvae"]),
    ("bvae_config", ["experiment=imnet256/bvae-skipvit-b-2"]),
    ("rf_config", []),
    ("rf_config", ["experiment=smoke/decoder"]),
    ("rf_config", ["experiment=imnet256/sit-xl-2_context_cond"]),
]


def iter_targets(node):
    if isinstance(node, dict):
        if isinstance(node.get("target"), str):
            yield node["target"]
        for value in node.values():
            yield from iter_targets(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_targets(value)


@pytest.mark.parametrize("config_name,overrides", RUNS, ids=lambda x: str(x))
def test_config_composes_and_targets_import(config_name, overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    container = OmegaConf.to_container(cfg, resolve=True)

    for target in sorted(set(iter_targets(container))):
        module_name, _, attr = target.rpartition(".")
        assert callable(getattr(importlib.import_module(module_name), attr)), target


@pytest.mark.parametrize("config_name", ["bvae_config", "rf_config"])
def test_data_and_trainer_share_timesteps(config_name):
    """The trainer reads ``batch[f"latents_{t:.2f}"]``, so data and trainer must agree on t."""
    overrides = ["experiment=imnet256/bvae-skipvit-b-2" if config_name == "bvae_config"
                 else "experiment=imnet256/sit-xl-2_context_cond", "source_timestep=0.2"]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    assert cfg.trainer_module.params.source_timestep == 0.2
    assert cfg.data.params.source_timestep == 0.2
    assert cfg.data.params.train.params.source_timestep == 0.2
    assert cfg.data.params.validation.params.source_timestep == 0.2
