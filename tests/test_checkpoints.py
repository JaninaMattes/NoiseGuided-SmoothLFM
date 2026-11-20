"""Checkpoints written under old module names keep loading, and the frozen β-VAE loads strictly.

The ``weights`` tests run against the trained thesis checkpoints when
``NGSLFM_THESIS_CKPTS`` points to a folder containing them (searched recursively for *.ckpt).
"""
import importlib
import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ldm.legacy import LEGACY_TARGETS, migrate_targets
from ldm.models.wrapper.pretrained_beta_vae import PretrainedBetaVAE
from scripts.make_dummy_checkpoint import make_dummy_checkpoint
from test_configs import iter_targets


def importable(target: str) -> bool:
    module_name, _, attr = target.rpartition(".")
    return callable(getattr(importlib.import_module(module_name), attr, None))


@pytest.mark.parametrize("new_target", sorted(set(LEGACY_TARGETS.values())))
def test_legacy_targets_map_to_existing_classes(new_target):
    assert importable(new_target)


def test_migrate_targets_rewrites_nested_targets_only():
    old = {"target": "ldm.trainer_bvae_ti2.TrainerModuleLatentBetaVae",
           "params": {"vae_cfg": {"target": "ldm.frameworks.beta_vae.bvae_model_t2i.BetaVAEModel"},
                      "name": "ldm.trainer_bvae_ti2.TrainerModuleLatentBetaVae", "list": [{"target": "x.Y"}]}}

    new = migrate_targets(OmegaConf.create(old))

    assert new["target"] == "ldm.trainer_beta_vae.TrainerModuleLatentBetaVae"
    assert new["params"]["vae_cfg"]["target"] == "ldm.frameworks.beta_vae.bvae_model.BetaVAEModel"
    assert new["params"]["name"] == old["params"]["name"]          # only `target` keys are migrated
    assert new["params"]["list"] == [{"target": "x.Y"}]


@pytest.fixture(scope="module")
def dummy_checkpoint(tmp_path_factory) -> Path:
    return make_dummy_checkpoint(tmp_path_factory.mktemp("ckpt") / "beta_vae.ckpt")


def test_pretrained_beta_vae_loads_frozen_weights(dummy_checkpoint):
    model = PretrainedBetaVAE(str(dummy_checkpoint))
    state = torch.load(dummy_checkpoint, weights_only=False)["state_dict"]

    assert not model.training and not any(p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[f"model.{key}"])


def test_pretrained_beta_vae_fails_loudly_on_mismatched_weights(dummy_checkpoint, tmp_path):
    checkpoint = torch.load(dummy_checkpoint, weights_only=False)
    checkpoint["state_dict"].pop(next(iter(checkpoint["state_dict"])))
    broken = tmp_path / "broken.ckpt"
    torch.save(checkpoint, broken)

    with pytest.raises(RuntimeError, match="Missing key"):
        PretrainedBetaVAE(str(broken))


def thesis_checkpoints() -> list[Path]:
    root = os.environ.get("NGSLFM_THESIS_CKPTS")
    return sorted(p for p in Path(root).rglob("*.ckpt") if p.exists()) if root else []


@pytest.mark.weights
@pytest.mark.skipif(not thesis_checkpoints(), reason="set NGSLFM_THESIS_CKPTS to the trained checkpoints")
@pytest.mark.parametrize("path", thesis_checkpoints(), ids=lambda p: p.name)
def test_thesis_checkpoint_loads(path):
    hparams = torch.load(path, map_location="cpu", weights_only=False, mmap=True)["hyper_parameters"]
    plain = {k: OmegaConf.to_container(v) if OmegaConf.is_config(v) else v for k, v in hparams.items()}

    targets = set(iter_targets(migrate_targets(plain)))
    assert targets and all(importable(t) for t in targets), sorted(t for t in targets if not importable(t))
    if "vae_cfg" in plain:                                          # β-VAE: weights load strictly
        PretrainedBetaVAE(str(path))
