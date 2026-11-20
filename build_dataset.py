"""Entry point for generating the guidance-free HDF5 dataset (see dataprocessing/build.py).

    python build_dataset.py                                   # thesis subset from $IMAGENET_DIR
    python build_dataset.py experiment=dataset/sample         # small real dataset from repository images
"""
import logging

import hydra
from omegaconf import DictConfig

from dataprocessing.build import build_dataset


@hydra.main(config_path="configs", config_name="build_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    build_dataset(cfg)


if __name__ == "__main__":
    main()
