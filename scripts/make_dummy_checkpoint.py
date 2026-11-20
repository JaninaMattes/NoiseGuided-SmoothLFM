"""Write a randomly initialised β-VAE checkpoint with the layout of a real training checkpoint.

It lets the self-guided flow decoder run end to end without trained weights:

    python scripts/make_dummy_checkpoint.py                    (make dummy-checkpoint)
    make train-decoder BVAE_CKPT=checkpoints/dummy/beta_vae_skip_vit_s2.ckpt EXP=smoke/flow_decoder_guided
"""
import argparse
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ldm.legacy import instantiate

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_OUTPUT = "checkpoints/dummy/beta_vae_skip_vit_s2.ckpt"


def make_dummy_checkpoint(output: Path, seed: int = 2025) -> Path:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train_beta_vae", overrides=["experiment=smoke/beta_vae"])
    torch.manual_seed(seed)
    model = instantiate(cfg.model)
    checkpoint = {
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
        "hyper_parameters": {"vae_cfg": OmegaConf.to_container(cfg.model, resolve=True)},
        "dummy": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    print(f"Wrote {make_dummy_checkpoint(parser.parse_args().output)}")
