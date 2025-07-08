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

def gen_data(bs):
    return {
        "image": torch.randn(bs, 3, 256, 256),
        "label": torch.randint(0, 1000, (bs,)),
        "latent": torch.randn(bs, 4, 32, 32),
    }


@hydra.main(config_path="../../configs", config_name="ae_vit_default_config", version_base=None)
def main(cfg: DictConfig):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """Setup Torch Profiling"""
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
    
    """ Setup dataloader """
    data = instantiate_from_config(cfg.data)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup(stage='fit')
        print("Data setup complete.")
        
        
    """ Setup model """
    module = instantiate_from_config(cfg.trainer_module)
    module.model.to(DEV)

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