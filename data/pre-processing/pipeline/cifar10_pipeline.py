from ldm.flow import FlowModel
from ldm.models.transformer.dit import DiT_models
from data_processing.sampler.data_handler import NumpyDataHandler, HDF5DatasetManager
import argparse
import os
import sys

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from itertools import islice
import einops

import torch

# Load jutil modules
from jutils.nn import AutoencoderKL
from jutils import instantiate_from_config
from jutils import exists, freeze, default
from jutils import ims_to_grid


project_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../'))
sys.path.append(project_root)

# Load custom modules


# Set precision to high
torch.set_float32_matmul_precision('high')


#############################################################
#                          Utils                           #
#############################################################

def img_to_grid(img, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and img.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        img = einops.rearrange(
            img, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        img = einops.rearrange(img, "b c h w -> " + to)
    return img


def un_normalize_img(img):
    """ Convert from [-1, 1] to [0, 255] """
    img = ((img * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return img


def normalize_img(img):
    """ Convert from [0, 255] to [-1, 1] """
    img = img.to(torch.float32) / 127.5 - 1
    return img


def show_samples(intermediates, split=4, save_to_file=None):
    """ Show samples """
    intermediates = dict(sorted(intermediates.items(), key=lambda x: float(
        x[0]), reverse=True))  # Sort by timestep
    ims = torch.stack(list(intermediates.values()), dim=1)
    ims = einops.rearrange(ims, "t b c h w -> (t b) c h w")
    ims = un_normalize_img(ims)
    ims_grid = ims_to_grid(ims, stack="row", split=split)

    plt.imshow(ims_grid.cpu().numpy())
    plt.title(r"Forward Diffusion $x_1 \rightarrow x_0$")
    plt.axis("off")
    if save_to_file:
        plt.savefig(save_to_file, bbox_inches='tight')
    plt.show()
    plt.close()


#############################################################
#                 Pipelione Latent Sampler                   #
#############################################################

class SampleProcessor:
    def __init__(
        self,
        selected_timesteps: list,                       # Timesteps to extract
        dataset_cfg: str,                               # Dataset config
        dataset_dir: str,                               # Dataset directory
        hdf5_dir: str,                                  # HDF5 directory
        # First stage   (KL Autoencoder)
        first_stage_ckpt='checkpoints/sd_ae.ckpt',
        # Second stage  (LDM using Flow Matching)#
        second_stage_ckpt='checkpoints/SiT-XL-2-256x256.pt',
        start_batch_id: int = 0,                        # Starting batch ID
        end_batch_id: int = 10000,                      # Ending batch ID
        input_size: int = 32,                           # Input size
        num_classes: int = 10,                          # Number of classes
        class_labels: list = None,                      # Class labels to filter
        batch_size: int = None,                         # Batch size
        num_steps: int = 100,                           # Number of steps
        sample_kwargs: dict = None,                     # DDIM sampling kwargs
        dev: torch.device = None,                       # Device to use for sampling
        # Type of sampling (sample or encode)
        type: str = "train",
        log_every: int = 1000,                                 # Log every n batches
    ):

        # Device settings
        self.device = dev if dev else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.start_batch_id = start_batch_id
        self.end_batch_id = end_batch_id
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.log_every = log_every

        # Dataset settings
        self.data_dir = dataset_dir
        self.hdf5_dir = hdf5_dir
        self.dataset_cfg = dataset_cfg
        self.type = type

        # Sampling settings
        y_null = torch.tensor([self.num_classes] *
                              self.batch_size, device=self.device)
        self.sample_kwargs = sample_kwargs if sample_kwargs else {}
        self.sample_kwargs.update(  # Add null
            num_steps=num_steps,
            cfg_scale=1.0,    # unconditional sampling
            uc_cond=y_null,
            cond_key='y'
        )
        self.selected_timesteps = selected_timesteps
        if not self.selected_timesteps:
            raise ValueError(
                "No timesteps provided. Please specify --timesteps.")

        # KL Autoencoder - first stage settings
        first_stage = AutoencoderKL(ckpt_path=first_stage_ckpt).to(self.device)
        self.first_stage = torch.compile(first_stage, fullgraph=True)
        freeze(self.first_stage)
        self.first_stage.eval()

        # Second stage (Flow with SiT) settings
        flow_model = net = DiT_models["DiT-XL/2"](
            input_size=self.input_size,
            num_classes=self.num_classes,
            learn_sigma=True,               # we "learn sigma" but never use it in SiT/DiT
            load_from_ckpt=second_stage_ckpt,
        ).to(self.device)
        flow_model = FlowModel(net, schedule="linear").to(self.device)
        self.second_stage = torch.compile(flow_model, fullgraph=True)
        freeze(self.second_stage)
        self.second_stage.eval()

        # Load data
        cfg_path = dataset_cfg
        dataset_cfg = OmegaConf.load(cfg_path)
        self.datamod = instantiate_from_config(dataset_cfg)
        assert class_labels is not None, "Please provide class labels to filter the dataset."

        self.datamod.setup('fit')

        """ Data handler """
        self.datahandler = NumpyDataHandler(base_dir=self.data_dir)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if exists(self.first_stage):
            x = self.first_stage.encode(x)
        return x

    @torch.no_grad()
    def decode_first_stage(self, z):
        if exists(self.first_stage):
            z = self.first_stage.decode(z)
        return z

    @torch.no_grad()
    def encode_second_stage(self, latent, y=None, return_intermediates=True, sample_kwargs=None):
        """ Forward diffusion """
        if exists(self.second_stage):
            xt, intermediates = self.second_stage.encode(latent, y=y, return_intermediates=return_intermediates, **(
                sample_kwargs or {}))               # x0: noise, x: target, t: timestep
        return xt, intermediates

    @torch.no_grad()
    def decode_second_stage(self, z, label=None):
        """ Euler sampling """
        if exists(self.second_stage):
            z = self.second_stage.generate(z, y=label, **self.sample_kwargs)
        return z

    @torch.no_grad()
    def __call__(self):
        """Generate noisy latents"""
        if self.type == 'train':
            dataloader = self.datamod.train_dataloader()
        else:
            dataloader = self.datamod.val_dataloader()

        # Get selected timesteps
        selected_timesteps = sorted(self.selected_timesteps, reverse=True)
        print(f"Selected timesteps: {selected_timesteps}")

        # Get batch
        for batch_idx, batch in enumerate(islice(dataloader, self.start_batch_id, self.end_batch_id), start=self.start_batch_id):
            if batch_idx >= self.end_batch_id:
                break

            x = batch['image'][:self.batch_size].to(self.device).float()
            y = batch['label'][:self.batch_size].to(self.device).long()

            # Pipeline
            latent = self.encode_first_stage(x)
            xt, intermediates = self.encode_second_stage(
                latent, y=y, return_intermediates=True, sample_kwargs=self.sample_kwargs)
            # Generate samples
            intermediates = {f"{t:.1f}": intermediates.get(
                f"{t:.1f}", None) for t in selected_timesteps}
            intermediates = {k: v for k,
                             v in intermediates.items() if v is not None}

            # Plot samples
            if batch_idx % self.log_every == 0:
                print(f"Batch {batch_idx}/{self.end_batch_id} - {self.type}")
                # img_file = os.path.join(self.data_dir, f"{self.type}_samples_{batch_idx}.png")
                # show_samples(intermediates, split=4, save_to_file=img_file)

            data_dict = {
                'image': x.detach().cpu(),
                'latent': xt.detach().cpu(),
                'label': y.detach().cpu(),
                'intermediate_steps': selected_timesteps,
                'intermediates': list(intermediates.values()),
            }

            # Save to Numpy
            self.datahandler.save_to_numpy(data_dict, group_name=self.type)
            torch.cuda.empty_cache()

        # Store to HDF5
        # postfix = datetime.datetime.now().strftime("T%H%M%S")
        # filename = f'imagenet256_data-{postfix}.hdf5'
        # filename = 'imagenet256-dataset-T000003.hdf5'
        # self.save_hdf5(self.data_dir, filename=filename, group_name=self.type)
        torch.cuda.empty_cache()

    def save_hdf5(self, data_dir, filename, group_name: str = 'train'):
        """ Save to HDF5 """
        hdfhandler = HDF5DatasetManager(data_dir)
        hdfhandler.save_to_hdf5(filename=filename, group_name=group_name)
        # Plot structure
        hdf5_file = os.path.join(data_dir, filename)
        hdfhandler.print_hdf5_structure(hdf5_file, save_to_file=True)
        print(f"Data saved to {hdf5_file}")

        # Load HDF5
        imgs, labels, latents = hdfhandler.retrieve_from_hdf5(
            file_path=hdf5_file, timestep=self.selected_timesteps[
                0], group_name=group_name, plot_samples=True
        )
        return imgs, labels, latents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latent Sample Generator for Diffusion Model Datasets")
    parser.add_argument('--dataset_dir', type=str,
                        default='dataset/processed/imagenet-256')
    parser.add_argument('--dataset_cfg', type=str,
                        default='configs/data/imagenet256_mvl.yaml')
    parser.add_argument('--timesteps', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument('--class_labels', type=int, nargs='+', default=[
        0, 1, 84, 87, 88, 89, 90, 92, 93, 94, 95, 96, 99, 100, 105, 106, 130, 144, 145, 152,
        153, 154, 158, 172, 176, 207, 208, 219, 231, 232, 234, 236, 237, 248, 249, 250, 251,
        254, 258, 259, 260, 263, 264, 269, 270, 271, 277, 278, 279, 280, 282, 283, 284, 288,
        289, 290, 291, 292, 293, 294, 295, 296, 321, 322, 323, 324, 330, 331, 332, 339, 340,
        344, 346, 347, 348, 349, 350, 352, 353, 354, 361, 362, 365, 366, 368, 383, 387, 388,
        954, 957
    ])  # V2 dataset class labels
    # Batch size for sampling
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_batch_id', type=int, default=0)
    parser.add_argument('--end_batch_id', type=int, default=10)
    parser.add_argument('--split', type=str,
                        choices=['train', 'validation'], default='train')
    parser.add_argument('--hdf5_file', type=str, default=None)
    parser.add_argument('--second_stage_ckpt', type=str,
                        default='checkpoints/SiT-XL-2-256x256.pt')

    args = parser.parse_args()

    print(f"Selected timesteps: {args.timesteps}")
    class_labels = sorted(args.class_labels)
    print(f"Class labels: {class_labels}")
    print(f"Number of class labels: {len(class_labels)}")

    processer = SampleProcessor(
        selected_timesteps=args.timesteps,
        dataset_cfg=args.dataset_cfg,
        dataset_dir=args.dataset_dir,
        hdf5_dir=args.hdf5_file,
        start_batch_id=args.start_batch_id,
        end_batch_id=args.end_batch_id,
        num_classes=1000,
        class_labels=class_labels,
        batch_size=args.batch_size if args.batch_size > 0 else len(
            class_labels),
        log_every=5000,
        type=args.split,
        filtered_loader=args.filtered_loader,
    )
    processer()
    print("Sample processing completed.")
    # processer.save_hdf5(args.dataset_dir, filename='imagenet256-dataset-T000003.hdf5', group_name=args.split)

    # CUDA_VISIBLE_DEVICES=0 python ... <- This line is a reminder to run the script with the appropriate CUDA device.
    # Note: The script is designed to run on a single GPU. If you want to run it on multiple GPUs, you need to modify the code to use DataParallel or DistributedDataParallel
