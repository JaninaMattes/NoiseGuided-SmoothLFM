# Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
# Code available at:
#    - https://github.com/gnobitab/RectifiedFlow/blob/main/ImageGeneration/sde_lib.py
#    - https://arxiv.org/pdf/2209.03003
#
# Code adapted from:
# - https://github.com/joh-schb/image-ldm/blob/main/ldm/flow.py
# - https://github.com/CompVis/fm-boosting/blob/main/fmboost/flow.py
# 
""" Adapted from
- https://github.com/atong01/conditional-flow-matching
- https://github.com/willisma/SiT
Thanks for open-sourcing! :)
"""
import math
import torch
import einops
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
from typing import Any, Dict, List, Optional, Union
from functools import partial
from torchdiffeq import odeint

from jutils import instantiate_from_config


# default from https://github.com/willisma/SiT
_ATOL = 1e-6
_RTOL = 1e-3


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


""" Schedules """


class LinearSchedule:
    def alpha_t(self, t):
        return t
    
    def alpha_dt_t(self, t):
        return 1
    
    def sigma_t(self, t):
        return 1 - t
    
    def sigma_dt_t(self, t):
        return -1

    """ Legacy functions to work with SiT Sampler """

    def compute_alpha_t(self, t):
        return self.alpha_t(t), self.alpha_dt_t(t)
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return self.sigma_t(t), self.sigma_dt_t(t)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t
    
    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = pad_v_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion
    
    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = pad_v_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }

        try: diffusion = choices[form]
        except KeyError: raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion
    
    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = pad_v_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity
    

class GVPSchedule(LinearSchedule):
    def alpha_t(self, t):
        return torch.sin(t * math.pi / 2)
    
    def alpha_dt_t(self, t):
        return 0.5 * math.pi * torch.cos(t * math.pi / 2)
    
    def sigma_t(self, t):
        return torch.cos(t * math.pi / 2)
    
    def sigma_dt_t(self, t):
        return - 0.5 * math.pi * torch.sin(t * math.pi / 2)
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * torch.tan(t * np.pi / 2))
    

""" Timestep Sampler """


class LogitNormalSampler:
    def __init__(self, loc: float = 0.0, scale: float = 1.0):
        """
        Logit-Normal sampler from the paper 'Scaling Rectified
        Flow Transformers for High-Resolution Image Synthesis'
        - Esser et al. (ICML 2024)
        """
        self.loc = loc
        self.scale = scale

    def __call__(self, n, device='cpu', dtype=torch.float32):
        return torch.sigmoid(self.loc + self.scale * torch.randn(n)).to(device).to(dtype)


""" Flow Model """


class FlowModel(nn.Module):
    def __init__(
            self,
            net_cfg: Union[dict, nn.Module],
            schedule: str = "linear",
            sigma_min: float = 0.0,
            timestep_sampler: dict = None,
        ):
        """
        Flow Matching, Stochastic Interpolants, or Rectified Flow model. :)
        
        Args:
            net: Neural network that takes in x and t and outputs the vector
                field at that point in time and space with the same shape as x.
            schedule: str, specifies the schedule for the flow. Currently
                supports "linear" and "gvp" (Generalized Variance Path) [3].
            sigma_min: a float representing the standard deviation of the
                Gaussian distribution around the mean of the probability
                path N(t * x1 + (1 - t) * x0, sigma), as used in [1].
            timestep_sampler: dict, configuration for the training timestep sampler.
        
        References:
            [1] Lipman et al. (2023). Flow Matching for Generative Modeling.
            [2] Tong et al. (2023). Improving and generalizing flow-based
                generative models with minibatch optimal transport.
            [3] Ma et al. (2024). SiT: Exploring flow and diffusion-based
                generative models with scalable interpolant transformers.
        """
        super().__init__()
        if isinstance(net_cfg, nn.Module):
            self.net = net_cfg
        else:
            self.net = instantiate_from_config(net_cfg)
        self.sigma_min = sigma_min

        if schedule == "linear":
            self.schedule = LinearSchedule()
        elif schedule == "gvp":
            assert sigma_min == 0.0, "GVP schedule does not support sigma_min."
            self.schedule = GVPSchedule()
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")
        
        if timestep_sampler is not None:
            self.t_sampler = instantiate_from_config(timestep_sampler)
        else:
            self.t_sampler = torch.rand             # default: uniform U(0, 1)

    def forward(self, x: Tensor, t: Tensor, cfg_scale=1.0, uc_cond=None, cond_key="y", **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        # _pred = self.net(x=x, t=t, **kwargs)
        _pred = forward_with_cfg(x, t, self.net, cfg_scale=cfg_scale, uc_cond=uc_cond, cond_key=cond_key, **kwargs)
        return _pred
    
    
    def get_indices(self, t: List[float], t_delta: float) -> List[int]:
        """ Get specific indices based on t_delta for evenly spaced t."""
        num_steps = len(t)
        step_size = int(num_steps * t_delta) 

        selected_indices = [i for i in range(-1, num_steps - 1, step_size)]
        selected_indices = sorted(set(selected_indices))
        return selected_indices
    
    
    def get_schedule(self,
        x: Tensor,
        num_steps: int,
        image_seq_len: int,
        t_start: int = 0,
        t_end: int = 1,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = False,
        reverse: bool = False
    ) -> List[float]:

        t = torch.linspace(t_start, t_end, num_steps, dtype=x.dtype).to(x.device)
        t = 1 - t if reverse else t

        # shifting the schedule to favor high t for higher signal images
        if shift:
            # estimate mu based on linear estimation between two points
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            t = self.time_shift(mu, 1.0, t)
        
        return t
    

    def encode_legacy(
        self,
        img: Tensor,
        t_start: float = 0,
        t_end: float=1,
        t_delta: float = 1.,
        ddim_steps: int = 100,
        n_intermediates: int = 0,
        image_seq_len: int = 64,
        reverse: bool = True,
        shift: bool = False,
        single_step: bool = False,
        sample_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode with classifier-free guidance support and flexible configuration.
    
        Args:
            x: Input tensor to encode
            sample_kwargs: Dictionary containing sampling parameters including:
                - num_steps: Number of diffusion steps
                - cfg_scale: Classifier-free guidance scale
                - uc_cond: Unconditional conditioning tensor (1, *dim) or (bs, *dim)
                - use_sde: Whether to use SDE sampling or ODE sampling
                - method: Sampling method (euler, heun, etc.)
                Additional parameters for ODE/SDE sampling can be included
            **kwargs: Additional arguments passed to the network
        """
        intermediates: List[Tensor] = []
        inter_steps: List[float] = []
        return_intermediates = n_intermediates > 0

        sample_kwargs = sample_kwargs or {}
        cfg_scale = sample_kwargs.get("cfg_scale", 4.0)

        # Define schedule
        t = self.get_schedule(img, ddim_steps, image_seq_len, t_start=t_start, t_end=t_end, shift=shift, reverse=reverse)
        if return_intermediates:
            save_indices = self.get_indices(t, t_delta=t_delta)
        else:
            save_indices = []

        # Get conditional and unconditional inputs
        y = kwargs.get(sample_kwargs.get("cond_key", "y"))
        if y is not None:
            if isinstance(y, int):
                y = torch.full((img.shape[0],), y, dtype=torch.long, device=img.device)
            elif isinstance(y, Tensor) and y.dim() == 0:
                y = y.expand(img.shape[0])
            
            # Null labels for unconditional conditioning
            y_null = sample_kwargs.get("uc_cond", torch.full_like(y, 1000))
        
        last_timestep = None
        for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), total=ddim_steps)):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            if y is not None:
                cond_pred = self.net(x=img, t=t_vec, **{**kwargs, sample_kwargs.get("cond_key", "y"): y})
                uncond_pred = self.net(x=img, t=t_vec, **{**kwargs, sample_kwargs.get("cond_key", "y"): y_null})
                pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            else:
                pred = self.net(x=img, t=t_vec, **kwargs)
            
            dt = t_prev - t_curr
            img = img + dt * pred

            if i in save_indices:
                intermediates.append(img.cpu())
                inter_steps.append(t_curr)

            last_timestep = t_curr
            
            if single_step:
                break
        
        out = {'x_encoded': img.cpu(), 'intermediate_steps': inter_steps, 'last_timestep': last_timestep}
        if return_intermediates:
            out.update({'intermediates': intermediates})

        return out
    
    
    def encode(self, x: Tensor, t_start: float = 0, t_end: float = 1, search_key: float = 0., num_steps: int = 50, reverse=True, return_intermediates=False, progress=True, **kwargs):
        """ Encoding process. With Euler sampling from x1 to x0 in num_steps.
        
        Args:
            x: source minibatch (bs, *dim)
            num_steps: int, number of steps to take
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: bool, if true, return list of samples
            progress: bool, if true, show tqdm progress bar
            kwargs: additional arguments for the network (e.g. conditioning information)
        """
        bs, dev = x.shape[0], x.device
        timesteps = torch.linspace(t_start, t_end, num_steps + 1)
        if reverse:
            timesteps = 1 - timesteps
        
        xt = x
        intermediates = dict()
        # Get intermediate steps
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            pred = self.forward(xt, t, **kwargs)

            dt = t_next - t_curr
            xt = xt + dt * pred

            if return_intermediates:                 
                key = f"{t_curr.item():.2}"
                intermediates[key] = xt
                if f"{search_key:.2}" == key:
                    break

        return xt, intermediates
        
        
        
    def decode(self,  x: Tensor, t_start: float = 0, t_end: float = 1, search_key: float = 1., num_steps: int = 50, reverse=False, return_intermediates=False, progress=True, **kwargs):
        """Decoding process. With Euler sampling from x0 to x1 in num_steps.
        
        Args:
            x: source minibatch (bs, *dim)
            num_steps: int, number of steps to take
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: bool, if true, return list of samples
            progress: bool, if true, show tqdm progress bar
            kwargs: additional arguments for the network (e.g. conditioning information
        """
        bs, dev = x.shape[0], x.device
        timesteps = torch.linspace(0, 1, num_steps + 1)
        if reverse:
            timesteps = 1 - timesteps

        xt = x
        intermediates = dict()
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            pred = self.forward(xt, t, **kwargs)

            dt = t_next - t_curr
            xt = xt + dt * pred

            if return_intermediates: 
                key = f"{t_curr.item():.4}"
                intermediates[key] = xt
                if f"{search_key:.4}" == key:
                    break

        return xt, intermediates

        

    def generate(self, x: Tensor, num_steps: int = 50, reverse=False, return_intermediates=False, progress=True, **kwargs):
        """
        Classic Euler sampling from x0 to x1 in num_steps.

        Args:
            x: source minibatch (bs, *dim)
            num_steps: int, number of steps to take
            reverse: bool, whether to reverse the direction of the flow. If True,
                we map from x1 -> x0, otherwise we map from x0 -> x1.
            return_intermediates: bool, if true, return list of samples
            progress: bool, if true, show tqdm progress bar
            kwargs: additional arguments for the network (e.g. conditioning information).
        """
        bs, dev = x.shape[0], x.device

        timesteps = torch.linspace(0, 1, num_steps + 1)
        if reverse:
            timesteps = 1 - timesteps

        xt = x
        intermediates = [xt]
        for t_curr, t_next in tqdm(zip(timesteps[:-1], timesteps[1:]), disable=not progress, total=len(timesteps)-1):
            t = torch.ones((bs,), dtype=x.dtype, device=dev) * t_curr
            pred = self.forward(xt, t, **kwargs)

            dt = t_next - t_curr
            xt = xt + dt * pred

            if return_intermediates: intermediates.append(xt)

        if return_intermediates:
            return torch.stack(intermediates, 0)
        return xt



    """ Training Helpers """
    def sample_xt(self, x0: Tensor, x1: Tensor, t: float=0.0):
        """ Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
            according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].
        """
        bs, dev = x0.shape[0], x0.device
        t = torch.full((bs,), t, dtype=x0.dtype, device=dev)
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt
    
    
    
    """ Training """

    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Sample from the time-dependent density p_t
            xt ~ N(alpha_t * x1 + sigma_t * x0, sigma_min * I),
        according to Eq. (1) in [3] and for the linear schedule Eq. (14) in [2].

        Args:
            x0 : shape (bs, *dim), represents the source minibatch (noise)
            x1 : shape (bs, *dim), represents the target minibatch (data)
            t  : shape (bs,) represents the time in [0, 1]
        Returns:
            xt : shape (bs, *dim), sampled point along the time-dependent density p_t
        """
        t = pad_v_like_x(t, x0)
        alpha_t = self.schedule.alpha_t(t)
        sigma_t = self.schedule.sigma_t(t)
        xt = alpha_t * x1 + sigma_t * x0
        if self.sigma_min > 0:
            xt += self.sigma_min * torch.randn_like(xt)
        return xt

    def compute_ut(self, x0: Tensor, x1: Tensor, t: Tensor):
        """
        Compute the time-dependent conditional vector field
            ut = alpha_dt_t * x1 + sigma_dt_t * x0,
        see Eq. (7) in [3].

        Args:
            x0 : Tensor, shape (bs, *dim), represents the source minibatch (noise)
            x1 : Tensor, shape (bs, *dim), represents the target minibatch (data)
            t  : FloatTensor, shape (bs,) represents the time in [0, 1]
        Returns:
            ut : conditional vector field
        """
        t = pad_v_like_x(t, x0)
        alpha_dt_t = self.schedule.alpha_dt_t(t)
        sigma_dt_t = self.schedule.sigma_dt_t(t)
        return alpha_dt_t * x1 + sigma_dt_t * x0

    def training_losses(self, x1: Tensor, x0: Tensor = None, **cond_kwargs):
        """
        Args:
            x1: shape (bs, *dim), represents the target minibatch (data)
            x0: shape (bs, *dim), represents the source minibatch, if None
                we sample x0 from a standard normal distribution.
            t: shape (bs,), represents the time in [0, 1]. If None, we sample
                according to the t_sampler (default: U(0, 1)).
            cond_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            loss: scalar, the training loss for the flow model
        """
        if x0 is None: x0 = torch.randn_like(x1)
        t = self.t_sampler(x1.shape[0], device=x1.device, dtype=x1.dtype)

        xt = self.compute_xt(x0=x0, x1=x1, t=t)
        ut = self.compute_ut(x0=x0, x1=x1, t=t)
        vt = self.forward(x=xt, t=t, **cond_kwargs)

        return (vt - ut).square().mean()

    def validation_losses(self, x1: Tensor, x0: Tensor = None, num_segments: int = 8, **cond_kwargs):
        """
        SD3 & Meta Movie Gen show that val loss correlates well with human quality. They
        compute the loss in equidistant segments in (0, 1) to reduce variance and average
        them afterwards. Default number of segments: 8 (Esser et al., page 21, ICML 2024).
        """
        if x0 is None: x0 = torch.randn_like(x1)

        assert num_segments > 0, "Number of segments must be greater than 0"
        ts = torch.linspace(0, 1, num_segments+1)[:-1] + 1/(2*num_segments)
        losses_per_segment = []
        for t in ts:
            t = torch.ones(x1.shape[0], device=x1.device) * t
            xt = self.compute_xt(x0=x0, x1=x1, t=t)
            ut = self.compute_ut(x0=x0, x1=x1, t=t)
            vt = self.forward(x=xt, t=t, **cond_kwargs)
            losses_per_segment.append((vt - ut).square().mean())
        
        losses_per_segment = torch.stack(losses_per_segment)
        return losses_per_segment.mean(), losses_per_segment