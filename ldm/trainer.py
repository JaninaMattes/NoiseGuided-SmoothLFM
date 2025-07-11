import wandb
import torch
import einops
from PIL import Image

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image.fid import FrechetInceptionDistance

from jutils import instantiate_from_config
from jutils import load_partial_from_config
from jutils import exists, freeze, default

from ldm.ema import EMA


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def ims_to_grid(ims, stack="row", split=4):
    """ Convert (b, c, h, w) to (h, w, c) """
    if stack not in ["row", "col"]:
        raise ValueError(f"Unknown stack type {stack}")
    if split is not None and ims.shape[0] % split == 0:
        splitter = dict(b1=split) if stack == "row" else dict(b2=split)
        ims = einops.rearrange(ims, "(b1 b2) c h w -> (b1 h) (b2 w) c", **splitter)
    else:
        to = "(b h) w c" if stack == "row" else "h (b w) c"
        ims = einops.rearrange(ims, "b c h w -> " + to)
    return ims


class TrainerModuleLatentFlow(LightningModule):
    def __init__(
        self,
        flow_cfg: dict,
        first_stage_cfg: dict = None,
        lr: float = 1e-4,
        weight_decay: float = 0.,
        n_images_to_vis: int = 16,
        ema_rate: float = 0.99,
        ema_update_every: int = 1,
        ema_update_after_step: int = 1,
        num_classes: int = 0,
        lr_scheduler_cfg: dict = None,
        log_grad_norm: bool = True,
        sample_kwargs: dict = None
    ):
        super().__init__()

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
        
        # first stage settings
        self.first_stage = None
        if exists(first_stage_cfg):
            first_stage = instantiate_from_config(first_stage_cfg)
            self.first_stage = torch.compile(first_stage, fullgraph=True)
            freeze(self.first_stage)
            self.first_stage.eval()

        # training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        # visualization
        self.sample_kwargs = sample_kwargs or {}
        self.n_images_to_vis = n_images_to_vis
        self.image_shape = None
        self.latent_shape = None
        self.generator = torch.Generator()
        self.num_classes = num_classes

        # evaluation
        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=True,
            normalize=False,
            sync_on_compute=True
        ).to(self.device)

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
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out
    
    @torch.no_grad()
    def encode(self, x):
        if exists(self.first_stage):
            x = self.first_stage.encode(x)
        return x
    
    @torch.no_grad()
    def decode(self, z):
        if exists(self.first_stage):
            z = self.first_stage.decode(z)
        return z
    
    def forward(self, x, latent=None, **kwargs):
        latent = default(latent, self.encode(x))
        loss = self.model.training_losses(latent, **kwargs)
        return loss
    
    def training_step(self, batch, batch_idx):
        ims = batch["image"]
        latent = default(batch.get("latent"), None)
        label = batch.get("label", None)
        
        loss = self.forward(ims, latent=latent, y=label)
        self.log("train/loss", loss, on_step=True, on_epoch=False, batch_size=ims.shape[0], sync_dist=True)

        if exists(self.ema_model): self.ema_model.update()
        if exists(self.lr_scheduler_cfg): self.lr_schedulers().step()
        if self.stop_training: self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        ims = batch["image"]
        label = batch.get("label", None)
        latent = default(batch.get("latent"), None)
        bs = ims.shape[0]
        
        # only flow models val loss shows correlation with human quality
        if hasattr(self.model, 'validation_losses'):
            latent = default(latent, self.encode(ims))
            _, val_loss_per_segment = self.model.validation_losses(latent, y=label)
            self.val_losses.append(val_loss_per_segment)

        # generation
        if self.latent_shape is None:
            _latent = self.encode(ims)
            self.latent_shape = _latent.shape[1:]
            self.batch_size = bs
        
        # sample images
        g = self.generator.manual_seed(batch_idx)
        z = torch.randn((bs, *self.latent_shape), generator=g).to(self.device)
        sampler = self.ema_model.model if exists(self.ema_model) else self.model
        samples = sampler.generate(x=z, y=label, **self.sample_kwargs)
        samples = self.decode(samples)
        
        # FID
        real_ims = un_normalize_ims(ims)
        fake_ims = un_normalize_ims(samples)
        self.fid.update(real_ims, real=True)
        self.fid.update(fake_ims, real=False)

    def on_validation_epoch_end(self):
        g = self.generator.manual_seed(2024)
        z = torch.randn((self.batch_size, *self.latent_shape), generator=g).to(self.device)

        # ignored in model for unconditional training
        if self.num_classes > 0:
            y = torch.randint(0, self.num_classes, (self.batch_size,), generator=g).to(self.device)
        else:
            y = None

        # sample
        samples = self.model.generate(x=z, y=y, num_steps=50)
        samples = samples[:self.n_images_to_vis]
        samples = self.decode(samples)
        samples = un_normalize_ims(samples)
        self.log_images(samples, "val/samples", stack="row", split=4)

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

        # log some information
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}: {fid:.2f} FID")

    def log_images(self, ims, name, stack="row", split=4):
        """
        Args:
            ims: torch.Tensor or np.ndarray of shape (b, c, h, w) in range [0, 255]
            name: str
        """
        ims = ims_to_grid(ims, stack=stack, split=split)
        if isinstance(ims, torch.Tensor):
            ims = ims.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            ims = Image.fromarray(ims)
            ims = wandb.Image(ims)
            self.logger.experiment.log({f"{name}/samples": ims})
        else:
            ims = einops.rearrange(ims, "h w c -> c h w")
            self.logger.experiment.add_image(f"{name}/samples", ims, global_step=self.global_step)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm