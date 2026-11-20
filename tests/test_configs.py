"""Every root/experiment config must compose and every ``target:`` in it must be importable.

Nothing is instantiated, so no data or checkpoints are needed. This guards against configs
pointing at modules or config groups that do not exist in the repository.
"""
import importlib

import pytest
from omegaconf import OmegaConf

from conftest import CONFIG_DIR, compose_config

RUNS = [
    ("train_beta_vae", []),
    ("train_beta_vae", ["experiment=smoke/beta_vae"]),
    ("train_flow_decoder", []),
    ("train_flow_decoder", ["experiment=smoke/flow_decoder"]),
    ("train_flow_decoder", ["experiment=smoke/flow_decoder_guided"]),
    ("build_dataset", []),
    ("build_dataset", ["experiment=smoke/dataset"]),
    ("build_dataset", ["experiment=dataset/sample"]),
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
    container = OmegaConf.to_container(compose_config(config_name, overrides), resolve=True)

    for target in sorted(set(iter_targets(container))):
        module_name, _, attr = target.rpartition(".")
        assert callable(getattr(importlib.import_module(module_name), attr)), target


def test_every_experiment_is_tested():
    """New experiment presets must be added to RUNS, otherwise they are never composed."""
    tested = {o.removeprefix("experiment=") for _, overrides in RUNS for o in overrides}
    experiments = {p.relative_to(CONFIG_DIR / "experiment").with_suffix("").as_posix()
                   for p in (CONFIG_DIR / "experiment").rglob("*.yaml")}
    assert experiments <= tested, f"add to RUNS: {sorted(experiments - tested)}"


@pytest.mark.parametrize("config_name", ["train_beta_vae", "train_flow_decoder"])
def test_data_and_trainer_share_timesteps(config_name):
    """The trainer reads ``batch[f"latents_{t:.2f}"]``, so data and trainer must agree on t."""
    cfg = compose_config(config_name, ["source_timestep=0.2"])

    assert cfg.trainer_module.params.source_timestep == 0.2
    assert cfg.data.params.source_timestep == 0.2
    assert cfg.trainer_module.params.target_timestep == cfg.data.params.target_timestep


@pytest.mark.parametrize("config_name", ["train_beta_vae", "train_flow_decoder"])
def test_training_reads_the_generated_dataset(config_name):
    """Training must read the file build_dataset.py writes, and every timestep it needs must be stored."""
    train, build = compose_config(config_name), compose_config("build_dataset")

    assert train.data.params.hdf5_file == build.output
    assert {train.source_timestep, train.target_timestep} <= set(build.timesteps)


def test_smoke_runs_share_the_smoke_dataset():
    build = compose_config("build_dataset", ["experiment=smoke/dataset"])
    for config_name, overrides in RUNS:
        if config_name != "build_dataset" and any(o.startswith("experiment=smoke/") for o in overrides):
            assert compose_config(config_name, overrides).data.params.hdf5_file == build.output
