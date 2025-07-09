import logging
import os
import sys


import os
import time
import logging
import os
import time
import logging
from omegaconf import OmegaConf
from itertools import islice
import torch

import torch
import torchvision
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import numpy as np
from omegaconf import OmegaConf

import webdataset as wds
from typing import Any, List, Optional


from jutils import instantiate_from_config


project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)


################################################################
#                   Webdatset Utilities                        #
################################################################
def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


def identity(x):
    return x


#####################################################
#       Filter for Micro-subsets in WebDataset      #
#####################################################

def keep_only_classes(sample, class_labels_set):
    class_sample = int(sample.get('cls', -1))
    return sample if class_sample in class_labels_set else None


def make_filtered_loader(
    data: Any,
    data_cfg: Any,
    class_labels: List[str],
    train: bool = True,
    batch_size: Optional[int] = None,
    num_batches: Optional[int] = None
) -> wds.WebLoader:
    """Create a filtered WebDataset loader."""
    tars = os.path.join(data.tar_base, data_cfg.shards)
    node_splitter = (wds.shardlists.split_by_node
                     if data.multinode
                     else wds.shardlists.single_node_only)

    dset_pipe = (
        wds.WebDataset(
            tars,
            shardshuffle=not data.multinode,
            nodesplitter=node_splitter,
            handler=wds.warn_and_continue
        )
        .repeat()
        .decode('rgb', handler=wds.warn_and_continue)
        .map(lambda sample: keep_only_classes(sample, set(class_labels)), handler=wds.warn_and_continue)
        .map(data.filter_out_keys, handler=wds.warn_and_continue)
    )

    if num_batches is not None:
        bs = data.batch_size if train else data.val_batch_size
        dset_pipe = dset_pipe.slice(num_batches * bs)
        logging.info(f"Limited dataset to {num_batches} batches.")

    image_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(
            lambda x: x * 2. - 1.)  # Normalize to [-1,1]
    ]

    if 'image_transforms' in data_cfg:
        custom_transforms = [
            instantiate_from_config(tt)
            for tt in data_cfg.image_transforms
        ]
        image_transforms.extend(custom_transforms)

    transform_dict = {
        data_cfg.image_key: torchvision.transforms.Compose(image_transforms)
    }
    dset_pipe = dset_pipe.map_dict(
        **transform_dict, handler=wds.warn_and_continue)

    if 'rename' in data_cfg:
        dset_pipe = dset_pipe.rename(**data_cfg.rename)

    bs = batch_size if batch_size is not None else (
        data.batch_size if train else data.val_batch_size)
    nw = data.num_workers if train else data.val_num_workers

    return wds.WebLoader(
        dset_pipe.batched(bs, partial=False, collation_fn=dict_collation_fn),
        batch_size=None,
        shuffle=False,
        num_workers=nw,
        prefetch_factor=2  # Overlap data loading with model training
    )


def visualize_batch(batch, class_names=None, save_path=None, nrow=8):
    """
    Visualizes a batch of images and optionally saves to disk.
    """
    images = batch['image']
    labels = batch['label']

    # Denormalize if needed (assumes [-1, 1] range from ToTensor + Lambda)
    images = (images + 1.0) / 2.0  # back to [0, 1]

    grid = vutils.make_grid(images[:nrow**2], nrow=nrow, padding=2)
    np_img = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 12))
    plt.imshow(np_img)
    plt.axis("off")

    # Plot labels as title
    if class_names:
        label_names = [class_names[int(l)] if isinstance(
            class_names, dict) else str(int(l)) for l in labels[:nrow**2]]
        plt.title("Labels: " + ", ".join(label_names), fontsize=10)
    else:
        label_ids = [str(int(l)) for l in labels[:nrow**2]]
        plt.title("Labels: " + ", ".join(label_ids), fontsize=10)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved image grid to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    #  Setup directories
    save_dir = "/tmp/filtered_batches"
    os.makedirs(save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info("WebDataset filtering test starting...")

    # Load configuration and dataset
    data_path = 'configs/data/imagenet256_mvl.yaml'
    dataset_cfg = OmegaConf.load(data_path)
    datamod = instantiate_from_config(dataset_cfg)
    data_cfg = datamod.train

    # Realistic subset of class labels
    test_class_labels = [
        0, 1, 84, 87, 88, 89, 90, 92, 93, 94, 95, 96, 99, 100, 105, 106, 130, 144, 145, 152, 153, 154, 158, 172,
        176, 207, 208, 219, 231, 232, 234, 236, 237, 248, 249, 250, 251, 254, 258, 259, 260, 263, 264, 269, 270,
        271, 277, 278, 280, 284, 288, 289, 290, 291, 292, 293, 294, 295, 296, 321, 322, 323, 324, 330, 331, 332,
        339, 340, 344, 346, 347, 348, 349, 350, 352, 353, 354, 361, 362, 365, 366, 368, 383, 387, 954, 957
    ]

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Time measurement start
    start_time = time.time()

    # Create filtered loader
    datamod.train_dataloader = make_filtered_loader(
        data=datamod,
        data_cfg=data_cfg,
        class_labels=test_class_labels,
        train=True,
        batch_size=32,
        num_batches=None  # Limit to 1000 batches
    )
    logging.info("Filtered loader created.")

    # Iterate through the batches
    sample_count = 0
    dataloader = datamod.train_dataloader
    start_batch_id = 0
    end_batch_id = 1000

    for batch_idx, batch in enumerate(islice(dataloader, start_batch_id, end_batch_id), start=start_batch_id):
        if batch_idx >= end_batch_id:
            break

        x = batch['image'][:32].to(device).float()
        y = batch['label'][:32].to(device).long()
        sample_count += x.size(0)

        # Save image grid
        # assuming [-1, 1] normalization
        grid = vutils.make_grid((x + 1) / 2.0, nrow=8)
        save_path = os.path.join(
            save_dir, f"batch_{batch_idx}_labels_{'_'.join(map(str, y.tolist()))}.png")
        vutils.save_image(grid, save_path)

        print(
            f"Batch {batch_idx}: x shape: {x.shape}, y shape: {y.shape} -> saved to {save_path}")

    # Timing end
    total_time = time.time() - start_time
    logging.info(
        f"Finished processing {sample_count} samples across {end_batch_id - start_batch_id} batches.")
    logging.info(f"Total time: {total_time:.2f} seconds.")
    logging.info(
        f"Avg time per batch: {total_time / (end_batch_id - start_batch_id):.4f} seconds.")
