from typing import Optional
import wandb
import torch
import einops
from PIL import Image
from functools import partial
import numpy as np

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance

from jutils import instantiate_from_config
from jutils import load_partial_from_config
from jutils import exists, freeze, default

from ldm.ema import EMA
from ldm.helpers import resize_ims, denorm_tensor


def un_normalize_img(img):
    """ Convert from [-1, 1] to [0, 255] """
    img = ((img * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return img


def img_to_grid(img, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and img.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        img = einops.rearrange(img, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        img = einops.rearrange(img, "b c h w -> " + to)
    return img


def latent2rgb(latents, projection=None):
    """
    Convert (B, 4, H, W) latents to (B, 3, H, W) using a learned or fixed 4x3 projection matrix.
    """
    B, C, H, W = latents.shape
    if projection is None:
        projection = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=latents.dtype, device=latents.device)  # shape (4, 3)

    latents_reshaped = latents.permute(0, 2, 3, 1)  # (B, H, W, 4)
    rgb = torch.matmul(latents_reshaped, projection)  # (B, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2)  # (B, 3, H, W)
    rgb = torch.clamp((rgb - rgb.min()) / (rgb.max() - rgb.min()), 0, 1)  # normalize to [0, 1]
    return rgb


def vis_taerget_pred_grid(target_img, pred_img, add_img=None, normalize=False):
    # resize pred_img if necessary
    if pred_img.shape[-1] != target_img.shape[-1]:
        pred_img = resize_ims(pred_img, target_img.shape[-1], mode="bilinear")
    target_img = einops.rearrange(target_img, "b c h w -> (b h) w c")
    pred_img = einops.rearrange(pred_img, "b c h w -> (b h) w c")
    if exists(add_img):
        add_img = einops.rearrange(add_img, "b c h w -> (b h) w c")
        grid = torch.cat([target_img, pred_img, add_img], dim=1)
    else:
        grid = torch.cat([target_img, pred_img], dim=1)
    # normalize to [0, 255] 
    if normalize:
        grid = un_normalize_img(grid)
    grid = grid.cpu().numpy()
    return grid


class TrainerModuleLatentFlow(LightningModule):
    def __init__(
        self,
        flow_cfg: dict,
        first_stage_cfg: Optional[dict] = None,                     # First stage   (KL Autoencoder)
        second_stage_cfg: Optional[dict] = None,                    # Second stage  (LDM using Flow Matching)
        third_stage_cfg: Optional[dict] = None,                     # Third stage   (Autoencoder pre-trained on LDM latent noised codes)
        metric_tracker_cfg: Optional[dict] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.,
        n_images_to_vis: int = 16,
        ema_rate: float = 0.99,
        ema_update_every: int = 1,
        ema_update_after_step: int = 1,
        num_classes: int = -1,                                      # -1: unconditional 
        source_timestep: float = 0.5,                               # (noise conditioning)  
        target_timestep: float = 1.0,                               # (data)
        lr_scheduler_cfg: Optional[dict] = None,
        sample_kwargs: Optional[dict] = None,
        use_context: bool = True,
        log_grad_norm: bool = False,
        optimizer: str = 'adamw',
    ):
        super().__init__()

        # Flow Model
        self.model = instantiate_from_config(flow_cfg)
        if ema_rate == 0.0:
            self.ema_model = None
        else:
            self.ema_model = EMA(
                self.model,
                beta=ema_rate,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
                power=3/4.,                     # recommended for trainings < 1M steps
                include_online_model=False      # we have the online model stored here
            )

        # Second stage (Flow with SiT) settings
        self.second_stage = None
        if exists(second_stage_cfg):
            flow_model = instantiate_from_config(second_stage_cfg)
            self.second_stage = torch.compile(flow_model, fullgraph=True)
            freeze(self.second_stage)
            self.second_stage.eval()

        # KL Autoencoder (SD3) - first stage settings
        self.first_stage = None
        if exists(first_stage_cfg):
            first_stage = instantiate_from_config(first_stage_cfg)
            self.first_stage = torch.compile(first_stage, fullgraph=True)
            freeze(self.first_stage)
            self.first_stage.eval()

        # LDM ß-VAE Autoencoder - third stage settings
        self.third_stage = None
        if exists(third_stage_cfg):
            third_stage = instantiate_from_config(third_stage_cfg)
            # TODO: Fix bug in compile function       
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                backend = 'inductor' if torch.cuda.get_device_capability()[0] >= 7 else 'aot_eager'
            else:
                backend = 'aot_eager'  # Default for CPU-only mode
            compile_fn = partial(torch.compile, fullgraph=True, backend=backend)
            self.third_stage = compile_fn(third_stage)
            freeze(self.third_stage)
                
            self.third_stage.eval()

        # Metric tracker
        self.metric_tracker = None
        if exists(metric_tracker_cfg):
            self.metric_tracker = instantiate_from_config(metric_tracker_cfg)
        else:
            self.fid = FrechetInceptionDistance(
                feature=2048,
                reset_real_features=True,
                normalize=False,
                sync_on_compute=True
            ).to(self.device)
            
        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        # visualization
        self.sample_kwargs = sample_kwargs or {}
        self.n_images_to_vis = n_images_to_vis
        self.image_shape = None
        self.latent_shape = None
        self.vis_samples = None
        self.log_every = 10
        self.generator = torch.Generator()
        self.num_classes = num_classes
        print(f"[INFO] Number of classes: {self.num_classes}")
        self.source_timestep = source_timestep
        self.target_timestep = target_timestep
        self.use_context = use_context

        # SD3 & Meta Movie Gen show that val loss correlates with human quality
        # and compute the loss in equidistant segments in (0, 1) to reduce variance
        self.val_losses = []        # only for Flow model
        
        self.val_epochs = 0
        self.save_hyperparameters()

        # signal handler for slurm, flag to make sure the signal
        # is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    # dummy function to be compatible
    def stop_training_method(self):
        pass

    def configure_optimizers(self):
        if self.optim == 'adamw':
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out
    
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
    def encode_second_stage(self, latent, t, y=None, sample_kwargs=None):
        sample_kwargs = sample_kwargs or {}
        if exists(self.second_stage):
            xt, _ = self.second_stage.encode(latent, search_key=t, y=y, **sample_kwargs)           # x0: noise, x: target, t: timestep
        return xt
    
    @torch.no_grad()
    def decode_second_stage(self, z, label=None, context=None, sample_kwargs=None):
        """ Euler sampling """
        sample_kwargs = sample_kwargs or {}
        if exists(self.second_stage):
            z = self.second_stage.generate(z, y=label, context=context, **sample_kwargs)
        return z
    
    @torch.no_grad()
    def encode_third_stage(self, x):
        if exists(self.third_stage):
            out = self.third_stage.encode(x)['latent_dist']
            x = out.sample() # Autoencoder trained to extract semantic context
        return x
    
    @torch.no_grad()
    def decode_third_stage(self, z):
        if exists(self.third_stage):
            z = self.third_stage.decode(z)['sample']
        return z
    
    def get_data_pair(self, img, x1_latent=None, xt_latent=None, label=None, use_context=False, sample_kwargs=None):
        """" Get training pair.
        
        Returns:
            x1: (bs, *dimg) target
            xt: (bs, *dimg) latent noise sample for semantic conditioning
            context: (bs, *c_dimg) extracted semantic context vector based on xtkl,mjm
        """
        context = None
        sample_kwargs = sample_kwargs or {}
        if not exists(x1_latent):
            x1_latent = self.encode_first_stage(img).to(torch.float32)                                                       # latent (x1): VAE encoded img
        if not exists(xt_latent):
            xt_latent = self.encode_second_stage(x1_latent, t=self.source_timestep, y=label, **sample_kwargs).to(torch.float32)                # latent (xt): LDM noise code        
        
        # Extract z-vector context
        if use_context:
            context = self.encode_third_stage(xt_latent).to(torch.float32)                                                  # Vector z context: extracted semantic context vector based on xt
        
        return dict(x1_latent=x1_latent, xt_latent=xt_latent, context=context)
            
            
    def forward(self, img, x1_latent=None, xt_latent=None, label=None, **kwargs):
        out = self.get_data_pair(img, x1_latent=x1_latent, xt_latent=xt_latent, label=label, 
                                 use_context=self.use_context, sample_kwargs=self.sample_kwargs) 
        x1_latent, context = out["x1_latent"], out["context"]
        loss = self.model.training_losses(x1=x1_latent, context=context, y=label, **kwargs)
        return loss
    
    
    def training_step(self, batch, batch_idx):
        img = batch["image"]
        label = batch["label"] if self.num_classes > 0 else None
        xt_latent = default(batch[f'latents_{self.source_timestep:.2f}'], None).to(torch.float32)          # Conditioning latent (xt): LDM noise code
        x1_latent = default(batch[f'latents_{self.target_timestep:.2f}'], None).to(torch.float32)          # X1 (data): VAE encoded img
        
        loss = self.forward(img, x1_latent=x1_latent, xt_latent=xt_latent, label=label)
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=img.shape[0], sync_dist=True)

        if exists(self.ema_model): self.ema_model.update()
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.stop_training: self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        label = batch["label"] if self.num_classes > 0 else None
        xt_latent = default(batch[f'latents_{self.source_timestep:.2f}'], None).to(torch.float32)          # Conditioning latent (xt): LDM noise code
        x1_latent = default(batch[f'latents_{self.target_timestep:.2f}'], None).to(torch.float32)          # X1 (data): VAE encoded img
        bs = img.shape[0]
        
        out = self.get_data_pair(img, x1_latent=x1_latent, xt_latent=xt_latent, label=label, 
                                 use_context=self.use_context, sample_kwargs=self.sample_kwargs) 
        x1_latent, xt_latent, context = out["x1_latent"], out["xt_latent"], out["context"]
        
        # only flow models val loss shows correlation with human quality
        if hasattr(self.model, 'validation_losses'):                
            _, val_loss_per_segment = self.model.validation_losses(x1_latent, context=context, y=label)
            self.val_losses.append(val_loss_per_segment)

        # generation
        if self.latent_shape is None:
            _latent = self.encode_first_stage(img)
            self.latent_shape = _latent.shape[1:]
            self.batch_size = bs
        
        g = self.generator.manual_seed(batch_idx)
        z = torch.randn((bs, *self.latent_shape), generator=g).to(self.device)
        sampler = self.ema_model.model if exists(self.ema_model) else self.model
        pred_samples = sampler.generate(x=z, y=label, context=context, **self.sample_kwargs)
        samples = self.decode_first_stage(pred_samples)
        
        # metrics
        real_img = un_normalize_img(img)
        fake_img = un_normalize_img(samples)
        
        if exists(self.metric_tracker):
            self.metric_tracker(real_img, fake_img)
        else:
            # FID
            self.fid.update(real_img, real=True)
            self.fid.update(fake_img, real=False)

        if batch_idx % self.log_every == 0:   
            
            # For visualization
            latent_gt = denorm_tensor(z)
            latent_target = denorm_tensor(x1_latent)
            latent_pred = denorm_tensor(pred_samples)
               
            if self.vis_samples is None:
                # create
                self.vis_samples = dict(
                    img_gt=real_img,
                    img_pred=fake_img,
                    latent_gt=latent_gt,
                    latent_target=latent_target,
                    latent_pred=latent_pred
                )
            elif self.vis_samples['img_gt'].shape[0] < self.n_images_to_vis:
                # append
                self.vis_samples['img_gt'] = torch.cat([self.vis_samples['img_gt'], real_img], dim=0)
                self.vis_samples['img_pred'] = torch.cat([self.vis_samples['img_pred'], fake_img], dim=0)
                self.vis_samples['latent_gt'] = torch.cat([self.vis_samples['latent_gt'], latent_gt], dim=0)
                self.vis_samples['latent_target'] = torch.cat([self.vis_samples['latent_target'], latent_target], dim=0)
                self.vis_samples['latent_pred'] = torch.cat([self.vis_samples['latent_pred'], latent_pred], dim=0)

        torch.cuda.empty_cache()
                

    def on_validation_epoch_end(self):
        g = self.generator.manual_seed(2025)
        z = torch.randn((self.batch_size, *self.latent_shape), generator=g).to(self.device)

        # ignored in model for unconditional training
        if self.num_classes > 0:
            y = torch.randint(0, self.num_classes, (self.batch_size,), generator=g).to(self.device)
        else:
            y = None
            print("[INFO] Unconditional training, no labels provided.")
    
        # sample
        context = None
        if self.use_context:
            context = default(context, self.encode_third_stage(z))
        
        samples = self.model.generate(x=z, y=y, context=context, num_steps=50)
        samples = samples[:self.n_images_to_vis]
        samples = self.decode_first_stage(samples)
        samples = un_normalize_img(samples)
        self.log_images(samples, "val/samples")

        # log metrics
        if exists(self.metric_tracker):
            metrics = self.metric_tracker.aggregate()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)
            self.metric_tracker.reset()
        else:
            # compute FID
            fid = self.fid.compute()
            self.log("val/fid", fid, sync_dist=True)
            self.fid.reset()

        # compute val loss if available (Flow models)
        if len(self.val_losses) > 0:
            val_losses = torch.stack(self.val_losses, 0)        # (N batches, segments)
            val_losses = val_losses.mean(0)                     # mean per segment
            for i, loss in enumerate(val_losses):
                self.log(f"val/loss_segment_{i}", loss, sync_dist=True)
            self.log("val/loss", val_losses.mean(), sync_dist=True)
            self.val_losses = []
        
        # Log samples
        if self.vis_samples is not None:
            out_imgs = vis_taerget_pred_grid(self.vis_samples["img_gt"], self.vis_samples["img_pred"])
            out_latents = vis_taerget_pred_grid(self.vis_samples["latent_gt"], self.vis_samples["latent_target"], self.vis_samples["latent_pred"])
            self.log_images(out_imgs, "samples/imgs", to_grid=False)
            self.log_images(out_latents, "samples/latents", to_grid=False)
            self.vis_samples = None
            
        # Update validation epoch count
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs} | Optimizer step {self.global_step}")

        torch.cuda.empty_cache()
        

    def log_images(self, img, name, stack="row", split=4, to_grid=True):
        """
        Args:
            img: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
            name: str
        """
        # resize if necessary
        if to_grid:
            img = img_to_grid(img, stack=stack, split=split)
        # convert to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # log to wandb
        if isinstance(self.logger, WandbLogger):
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = wandb.Image(img)
            self.logger.experiment.log({f"{name}/samples": img})
        else:
            img = einops.rearrange(img, "h w c -> c h w")
            self.logger.experiment.add_image(f"{name}/samples", img, global_step=self.global_step)

            
            
def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm