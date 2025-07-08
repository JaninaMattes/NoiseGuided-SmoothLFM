""" Adapted from
- https://github.com/atong01/conditional-flow-matching
- https://github.com/willisma/SiT
Thanks for open-sourcing! :)
"""
import torch
import einops


def pad_v_like_x(v_, x_):
    """
    Function to reshape the vector by the number of dimensions
    of x. E.g. x (bs, c, h, w), v (bs) -> v (bs, 1, 1, 1).
    """
    if isinstance(v_, float):
        return v_
    return v_.reshape(-1, *([1] * (x_.ndim - 1)))


def forward_with_cfg(x, t, model, cfg_scale=1.0, uc_cond=None, cond_key="y", context_key="context", **model_kwargs):
    """ Function to include sampling with Classifier-Free Guidance (CFG) """
    
    if cfg_scale == 1.0:                                # without CFG
        model_output = model(x, t, **model_kwargs)
        
    else:                                               # with CFG
        assert cond_key in model_kwargs, f"Condition key '{cond_key}' for CFG not found in model_kwargs"
        assert uc_cond is not None, "Unconditional condition not provided for CFG"
        kwargs = model_kwargs.copy()
        c = kwargs[cond_key]
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        if uc_cond.shape[0] == 1:
            uc_cond = einops.repeat(uc_cond, '1 ... -> bs ...', bs=x.shape[0])
        
        c_in = torch.cat([uc_cond, c])
        kwargs[cond_key] = c_in
        if context_key in kwargs and kwargs[context_key] is not None:
            context = kwargs[context_key]
            kwargs[context_key] = torch.cat([context] * 2, dim=0)
        
        model_uc, model_c = model(x_in, t_in, **kwargs).chunk(2)
        model_output = model_uc + cfg_scale * (model_c - model_uc)

    return model_output

