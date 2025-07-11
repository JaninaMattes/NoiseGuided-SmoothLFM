import wandb
import einops
from PIL import Image
from typing import Optional
from omegaconf import OmegaConf

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance

from jutils import instantiate_from_config
from jutils import load_partial_from_config
from jutils import exists, freeze, default

from ldm.ema import EMA
from ldm.helpers import get_batch_stats, resize_ims, denorm_tensor


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


class TrainerModuleLatentBetaVae(LightningModule):
    def __init__(
        self,
        vae_cfg: dict,
        first_stage_cfg: Optional[dict] = None,                     # First stage   (KL Autoencoder)
        second_stage_cfg: Optional[dict] = None,                    # Second stage  (LDM using Flow Matching)
        metric_tracker_cfg: Optional[dict] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.,
        n_images_to_vis: int = 16,
        batch_size: int = 16,
        ema_rate: float = 0.99,
        ema_update_every: int = 1,
        ema_update_after_step: int = 1,
        num_classes: int = 0,
        source_timestep: float = 0.50,
        target_timestep: float = 0.50,
        lr_scheduler_cfg: dict = None,
        log_grad_norm: bool = False,
        sample_kwargs: dict = None,
        optimizer: str = 'adamw',
    ):
        super().__init__()
        
        # Beta-VAE Model
        self.model = instantiate_from_config(vae_cfg)
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

        # KL Autoencoder - first stage settings
        self.first_stage = None
        if exists(first_stage_cfg):
            first_stage = instantiate_from_config(first_stage_cfg)
            self.first_stage = torch.compile(first_stage, fullgraph=True)
            freeze(self.first_stage)
            self.first_stage.eval()
            
        # Second stage (Flow with SiT) settings
        self.second_stage = None
        if exists(second_stage_cfg):
            flow_model = instantiate_from_config(second_stage_cfg)
            self.second_stage = torch.compile(flow_model, fullgraph=True)
            freeze(self.second_stage)
            self.second_stage.eval()

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
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.source_timestep = source_timestep               # xt (source): timestep for extraction of features from noise codes
        self.target_timestep = target_timestep               # xt (target): timestep for reconstruction target
        
        y_null = torch.tensor([self.num_classes] * self.batch_size, device=self.device)
        self.sample_kwargs = OmegaConf.to_container(sample_kwargs, resolve=True) 
        self.sample_kwargs.update(  # Add null
            uc_cond = y_null,
        )
        # visualization            
        self.n_images_to_vis = n_images_to_vis
        self.image_shape = None
        self.latent_shape = None
        self.vis_samples = None
        self.log_every = 10
        self.generator = torch.Generator()

        # Beta-Vae loss
        self.val_losses = None
        
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
    def encode_second_stage(self, latent, t, y=None):
        if exists(self.second_stage):
            xt, _ = self.second_stage.encode(latent, search_key=t, y=y, **self.sample_kwargs)               # x0: noise, x: target, t: timestep
        return xt
    
    @torch.no_grad()
    def decode_second_stage(self, z, label=None):
        """ Euler sampling """
        if exists(self.second_stage):
            z = self.second_stage.generate(z, y=label, **self.sample_kwargs)
        return z

            
    def forward(self, img, xt_source, xt_target, target_t=1.0, source_t=0.0, label=None):
        """ Forward pass """       
        return self.model(xt_source, xt_target)       
        
                
    def training_step(self, batch, batch_idx):
        img = batch["image"]
        label = batch["label"] if self.num_classes > 0 else None
        xt_source = default(batch.get(f'latents_{self.source_timestep:.2f}'), None)          # Conditioning latent (xt): LDM noise code
        xt_target = default(batch.get(f'latents_{self.target_timestep:.2f}'), None)          # X1 (data): VAE encoded img
        
        # get data pair
        out = self.forward(img, xt_source=xt_source, xt_target=xt_target, target_t=self.target_timestep, source_t=self.source_timestep, label=label)
        loss, recon_loss, kld_loss = out['loss'], out['recon_loss'], out['kld_loss']
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=img.shape[0], sync_dist=True)
        self.log("train/recon_loss", recon_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/kld_loss", kld_loss, on_step=True, on_epoch=False, sync_dist=True)

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
        xt_source = default(batch.get(f'latents_{self.source_timestep:.2f}'), None)          # Conditioning latent (xt): LDM noise code
        xt_target = default(batch.get(f'latents_{self.target_timestep:.2f}'), None)          # X1 (data): VAE encoded img
        
        # generation
        if self.latent_shape is None:
            bs = img.shape[0]
            latent = xt_source if exists(xt_source) else self.encode_first_stage(img)
            
            if self.model.use_vae:
                dist = self.model.encode(latent)['latent_dist']
                _latent = dist.sample()  # stochastic encoding
            else:
                _latent = self.model.encode(latent)['latent_dist']  # deterministic encoding

            self.latent_shape = _latent.shape[1:]
            self.batch_size = bs
            
        sampler = self.ema_model.model if exists(self.ema_model) else self.model
        out = sampler(xt_source, xt_target)
        loss, recon_loss, kld_loss, recon_sample = out['loss'], out['recon_loss'], out['kld_loss'], out['sample']
        samples = self.decode_second_stage(recon_sample, label=label)
        samples = self.decode_first_stage(samples)
        
        if self.val_losses is None:
            self.val_losses = {
                'loss': loss.unsqueeze(0),
                'recon_loss': recon_loss.unsqueeze(0),
                'kld_loss': kld_loss.unsqueeze(0),
            }
        else:
            self.val_losses['loss'] = torch.cat([self.val_losses['loss'], loss.unsqueeze(0)], dim=0)
            self.val_losses['recon_loss'] = torch.cat([self.val_losses['recon_loss'], recon_loss.unsqueeze(0)], dim=0)
            self.val_losses['kld_loss'] = torch.cat([self.val_losses['kld_loss'], kld_loss.unsqueeze(0)], dim=0)

        # metrics
        real_img = un_normalize_img(img)
        fake_img = un_normalize_img(samples)
        if exists(self.metric_tracker):
            self.metric_tracker(real_img, fake_img, xt_source, recon_sample)
        else:
            # FID
            self.fid.update(real_img, real=True)
            self.fid.update(fake_img, real=False)

        if batch_idx % self.log_every == 0:
            # Stats
            input_stats = get_batch_stats(xt_source)
            recon_stats = get_batch_stats(recon_sample)
            for k, v in input_stats.items():
                self.log(f"stats/input_{k}", v, sync_dist=True)
                self.log(f"stats/recon_{k}", recon_stats[k], sync_dist=True)
            
            # Visualization
            latent_train = denorm_tensor(xt_source)
            latent_gt = denorm_tensor(xt_target)
            latent_pred = denorm_tensor(recon_sample)
            if self.vis_samples is None:
                # create
                self.vis_samples = dict(
                    img_gt=real_img,
                    img_pred=fake_img,
                    latent_gt=latent_gt,
                    latent_train=latent_train,
                    latent_pred=latent_pred
                )
            elif self.vis_samples['img_gt'].shape[0] < self.n_images_to_vis:
                # append
                self.vis_samples['img_gt'] = torch.cat([self.vis_samples['img_gt'], real_img], dim=0)
                self.vis_samples['img_pred'] = torch.cat([self.vis_samples['img_pred'], fake_img], dim=0)
                self.vis_samples['latent_gt'] = torch.cat([self.vis_samples['latent_gt'], latent_gt], dim=0)
                self.vis_samples['latent_train'] = torch.cat([self.vis_samples['latent_train'], latent_train], dim=0)
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
    
        # sample    
        recon_sample = self.model.decode(z)["sample"]
        samples = self.decode_second_stage(recon_sample, label=y)
        samples = self.decode_first_stage(samples)
        samples = samples[:self.n_images_to_vis]
        samples = un_normalize_img(samples)
        self.log_images(samples, "val/samples", stack="row", split=4)

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

        # compute val loss 
        if self.val_losses['loss'].shape[0] > 0:
            for k, v in self.val_losses.items():
                self.log(f"val/{k}", v.mean(), sync_dist=True)
            self.val_losses = None  # reset
        
        # Log samples
        if self.vis_samples is not None:
            out_imgs = vis_taerget_pred_grid(self.vis_samples["img_gt"], self.vis_samples["img_pred"])
            out_latents = vis_taerget_pred_grid(self.vis_samples["latent_gt"], self.vis_samples["latent_train"], self.vis_samples["latent_pred"])
            self.log_images(out_imgs, "samples/imgs", to_grid=False)
            self.log_images(out_latents, "samples/latents", to_grid=False)
            self.vis_samples = None
            
        # Update validation epoch count
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs} | Optimizer step {self.global_step}")
        
        # Query Beta value (if annealer is used)
        beta = self.model.query_beta()  
        if exists(beta): 
            self.log("beta/annealed-beta", beta, sync_dist=True)
            
        torch.cuda.empty_cache()
        
        
    def log_images(self, img, name, stack="row", split=4, to_grid=True):
        """
        Args:
            img: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
            name: str
        """
        if to_grid:
            img = img_to_grid(img, stack=stack, split=split)
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
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