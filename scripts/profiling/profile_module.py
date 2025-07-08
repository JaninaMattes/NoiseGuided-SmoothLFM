import os
import sys
import hydra
import torch
from tqdm import tqdm
from functools import partial
from omegaconf import DictConfig

from torch.profiler import profile, ProfilerActivity, record_function

from jutils import NullObject
from jutils import instantiate_from_config

cdir = os.path.dirname(__file__)
pdir = os.path.dirname(cdir)
ppdir = os.path.dirname(pdir)
sys.path.insert(0, ppdir)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profile_fn = partial(
        profile,
        activities=[
            *((ProfilerActivity.CPU,) if cfg.profiling.cpu else ()),
            *((ProfilerActivity.CUDA,) if cfg.profiling.cuda else ()),
        ],
        record_shapes=cfg.profiling.record_shapes,
        profile_memory=cfg.profiling.profile_memory,
        with_flops=cfg.profiling.with_flops,
        with_stack=True,
    )

    """ Setup data """
    data = instantiate_from_config(cfg.data)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup()

    """ Setup model """
    module = TrainerModuleLatentAE(
        ae_cfg=cfg.model.ae_cfg, 
        autoencoder_cfg=cfg.get("autoencoder", None),
        flow_cfg=cfg.get("flow", None),
        scale_factor=cfg.train.get("scale_factor", 1.0),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        beta1_coeff=cfg.train.get("beta1", 0.9),
        beta2_coeff=cfg.train.get("beta2", 0.999),
        optimizer_type=cfg.train.get("optimizer", "adam"),
        batch_size=cfg.data.params.get("batch_size", 16),
        n_images_to_vis=cfg.train.get("n_images_to_vis", 16),
        ema_rate=cfg.train.get("ema_rate", 0.99),
        ema_update_after_step=cfg.train.get("ema_update_after_step", 1000),
        ema_update_every=cfg.train.get("ema_update_every", 100),
        num_classes=cfg.data.get("num_classes", 0),
        class_labels=cfg.data.get("class_labels", None),
        timestep=cfg.data.params.get("timestep", 0),
        t_delta=cfg.ddim.get("t_delta", 0.1),
        ddim_steps=cfg.ddim.get("steps", 100),
        n_intermediates=cfg.ddim.get("n_intermediates", 10),
        lr_scheduler_cfg=cfg.train.get("lr_scheduler", None),
        log_grad_norm=cfg.train.get("log_grad_norm", False),
        normalize=cfg.data.get("normalize", False),
    )
    

    """ Optimizer """
    opt = torch.optim.AdamW(module.parameters(), lr=cfg.trainer_module.params.lr)

    """ Run loop """
    profile_step = cfg.profiling.warmup
    for step, batch in enumerate(tqdm(data.train_dataloader(), total=profile_step)):

        batch = {k: v.to(DEV) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with profile_fn() if step == profile_step else NullObject() as prof:

            # forward pass
            with record_function(f"step_{step}/training_step"):
                loss = module.training_step(batch, step)
            
            # backward pass
            with record_function(f"step_{step}/backward"):
                module.backward(loss)

            # optimizer step
            with record_function(f"step_{step}/optimizer_step"):
                opt.step()
                opt.zero_grad()

        if step == profile_step:
            fn = f"profile_{module.__class__.__name__}_step{step}.json"
            print(f"[Profiling] Enabled after {cfg.profiling.warmup} steps.")
            print(f"[Profiling] Exporting {fn}")
            prof.export_chrome_trace(fn)
            break


if __name__ == "__main__":
    main()
