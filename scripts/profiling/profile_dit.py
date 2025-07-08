import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from jutils import NullObject
from functools import partial
from torch.profiler import profile, ProfilerActivity, record_function

parentdir = os.path.dirname(os.path.dirname(__file__))
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir)

from ldm.models.transformer.dit import DiT_models


def gen_data(bs):
    return {
        "x": torch.randn(bs, 4, 32, 32),
        "t": torch.rand((bs,)),
        "y": torch.randint(0, 1000, (bs,)),
    }


def main(args):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profile_fn = partial(
        profile,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_flops=False,
        with_stack=True,
    )

    """ Setup model """
    model = DiT_models["DiT-XL/2"]().to(DEV)

    # ===================================================== SETTINGS
    full_precision = False
    inf_context = NullObject() if full_precision else torch.autocast("cuda")
    # model = torch.compile(model, fullgraph=True)
    # ===================================================== SETTINGS

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    """ Run loop """
    profile_step = args.warmup + 1
    fn = f"dit_step{profile_step}_bs{args.bs}_{args.name}"
    for step in tqdm(range(profile_step), desc="Profiling"):
        
        batch = gen_data(args.bs)
        batch = {k: v.to(DEV) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with inf_context:
            with profile_fn() if step == args.warmup else NullObject() as prof:
                
                # forward pass
                with record_function(f"step_{step}/fwd"):
                    out = model(**batch)

                # backward pass
                with record_function(f"step_{step}/bwd"):
                    loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
                    loss.backward()

                # optimizer step
                with record_function(f"step_{step}/opt"):
                    opt.zero_grad()
                    opt.step()

        if not isinstance(prof, NullObject):
            print(f"[Profiling] Enabled after {profile_step} steps.")
            print(f"[Profiling] Exporting {fn}")
            prof.export_chrome_trace(f"{fn}.json")

    """ check timing """
    times = dict(fwd=[], bwd=[], opt=[], total=[])
    for step in tqdm(range(args.warmup + args.timing_steps), desc="Timing"):
        
        batch = gen_data(args.bs)
        batch = {k: v.to(DEV) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with inf_context:
            t0 = time.time()
            out = model(**batch)
            t_fwd = time.time()
            loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
            loss.backward()
            t_bwd = time.time()
            opt.zero_grad()
            opt.step()
            t_opt = time.time()

        if step < args.warmup:
            continue

        times['fwd'].append(t_fwd - t0)
        times['bwd'].append(t_bwd - t_fwd)
        times['opt'].append(t_opt - t_bwd)
        times['total'].append(t_opt - t0)
    
    print(f"Evaluated {len(times['fwd'])} steps for timing")
    with open(f"{fn}.txt", "w") as f:
        header = f"{'Process':<10} {'Mean':<10} {'Min':<10} {'Max':<10}"
        f.write(header + "\n")
        print(header)
        for k, v in times.items():
            v = np.array(v)
            txt = f"{k:<10} {v.mean():<10.4f} {v.min():<10.4f} {v.max():<10.4f}"
            f.write(txt + "\n")
            print(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--name", type=str, default="dit")
    parser.add_argument("--timing_steps", type=int, default=20)
    args = parser.parse_args()
    main(args)
