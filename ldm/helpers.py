# Code adapted from:
# - https://github.com/CompVis/fm-boosting/blob/main/fmboost/helpers.py
from functools import partial
import importlib
from inspect import isfunction
import einops
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


""" Helpers """
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


""" Model Loading """

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_partial_from_config(config):
    return partial(get_obj_from_str(config['target']),**config.get('params',dict()))


def load_model_from_config(config, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model


def load_model_weights(model, ckpt_path, strict=True, verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print("-" * 40)
        print(f"{'Loading weights':<16}: {ckpt_path}")
        print(f"{'Model':<16}: {model.__class__.__name__}")
        if "global_step" in ckpt:
            print(f"{f'Global Step':<16}: {ckpt['global_step']:,}")
        print(f"{'Strict':<16}: {'True' if strict else 'False'}")
        print("-" * 40)
    sd = ckpt["state_dict"] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if len(missing) > 0:
        warnings.warn(f"Load model weights - missing keys: {len(missing)}")
        if verbose:
            print(missing)
    if len(unexpected) > 0:
        warnings.warn(f"Load model weights - unexpected keys: {len(unexpected)}")
        if verbose:
            print(unexpected)
    return model


def load_model_from_ckpt(model, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model



def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


""" Batch Stats """

def get_batch_stats(x: torch.Tensor):
    """ Get stats of a mini batch of images """
    x = x.float()
    min = torch.min(x).item()
    max = torch.max(x).item()
    mean = torch.mean(x).item()
    std = torch.std(x).item()
    return dict(min=min, max=max, mean=mean, std=std)


""" Plotting """

def resize_ims(x: torch.Tensor, size: int, mode: str = "bilinear", **kwargs):
    # for the sake of backward compatibility
    if mode in ["conv", "noise_upsampling", "decoder_features"]:
        return nn.functional.interpolate(x, size=size, mode="bilinear", **kwargs)
    # idea: blur image before down-sampling
    return nn.functional.interpolate(x, size=size, mode=mode, **kwargs)


def plot_hist(predicted, original=None, bins=100):
    # matplotlib
    fig = plt.figure(figsize=(10, 7.5))
    if original is not None:
        plt.hist(original.detach().cpu().numpy().flatten(), bins=bins,
                alpha=0.5, label="original", density=True)
    plt.hist(predicted.detach().cpu().numpy().flatten(), bins=bins,
            alpha=0.5, label="predicted", density=True)
    plt.legend()
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title(f"bs={predicted.shape[0]} | dims={tuple(predicted.shape[1:])}")
    plt.grid()
    return fig


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





def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims



def denorm_tensor(tensor, min=0, max=255, keep_channels=3):
    """Denormalize a tensor with multiple channels for visualization.
    Returns:
        ims: tensor converted to [0, 255]
    """
    tensor = tensor.to(torch.float32)
    
    if tensor.size(1) > keep_channels:
        tensor = tensor[:, :keep_channels]  
        
    orig_min = tensor.amin(dim=(2, 3), keepdim=True)
    orig_max = tensor.amax(dim=(2, 3), keepdim=True)
    range_scale = max - min
    
    # Handle edge case
    scale = torch.where(orig_max > orig_min, range_scale / (orig_max - orig_min), torch.tensor(1.0, device=tensor.device))
    offset = min - orig_min * scale
    x = tensor * scale + offset
    x = torch.clamp(x, min, max)
    ims = x.round().to(torch.uint8)
    return ims


if __name__ == "__main__":
    # Example usage
    seed_everything(42)
    print("Seed set to 42")
    
    
    # Example tensor
    tensor = torch.randn(2, 3, 64, 64)
    denormed = denorm_tensor(tensor)
    print("Denormalized tensor shape:", denormed.shape)
    print(f"Min: {denormed.min()}, Max: {denormed.max()}")
    
    denormed = denorm_tensor(tensor, min=0, max=1, keep_channels=3)
    print("Denormalized tensor with custom range shape:", denormed.shape)
    print(f"Min: {denormed.min()}, Max: {denormed.max()}")
    
    
    # convert tensor in range -3.5, 3.0 to range 0, 255
    tensor = torch.randn(2, 3, 64, 64) * 3.5 + 3.0
    tensor = denorm_tensor(tensor, min=0, max=255)
    print("Denormalized tensor in range -3.5, 3.0 to 0, 255 shape:", tensor.shape)
    print(f"Min: {tensor.min()}, Max: {tensor.max()}")
    
    # convert tensor in range -3.5, 3.0 to range 0, 1
    tensor = torch.randn(2, 3, 64, 64) * 3.5 + 3.0
    tensor = denorm_tensor(tensor, min=0, max=1, keep_channels=3)
    print("Denormalized tensor in range -3.5, 3.0 to 0, 1 shape:", tensor.shape)
    print(f"Min: {tensor.min()}, Max: {tensor.max()}")
    
    # Plotting example
    fig = plot_hist(tensor[0])
    plt.show()
    print("Histogram plotted")