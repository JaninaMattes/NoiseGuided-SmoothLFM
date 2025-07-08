""" Adapted from
- https://github.com/atong01/conditional-flow-matching
- https://github.com/willisma/SiT
Thanks for open-sourcing! :)
"""
import torch
import einops
from tqdm import tqdm
from functools import partial
from torchdiffeq import odeint
from tqdm import tqdm

from ldm.frameworks.flow.schedule import GVPSchedule, LinearSchedule
from ldm.frameworks.flow.utils import forward_with_cfg


# default from https://github.com/willisma/SiT
_ATOL = 1e-6
_RTOL = 1e-3



""" SDE Sampler """


class StepSDE:
    """SDE solver class"""
    def __init__(self, dt, drift, diffusion, sampler_type):
        self.dt = dt
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.sampler_dict = {
            "euler": self.__Euler_Maruyama_step,
            "heun": self.__Heun_step,
        }

        try: self.sampler = self.sampler_dict[sampler_type]
        except: raise NotImplementedError(f"Sampler type '{sampler_type}' not implemented.")

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        t = torch.ones(x.size(0)).to(x) * t
        dw = w_cur * torch.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + torch.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size()).to(x)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __call__(self, x, mean_x, t, model, **model_kwargs):
        return self.sampler(x, mean_x, t, model, **model_kwargs)
    

class FlowSDE:
    def __init__(self, schedule, sample_eps=0):
        """ Sampler class for the FlowModel """
        self.sample_eps = sample_eps        # velocity & [GVP, LINEAR] is stable everywhere, hence 0
        self.schedule = schedule

    def drift(self, x, t, model, **model_kwargs):
        model_output = model(x, t, **model_kwargs)
        assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
        return model_output
    
    def score(self, x, t, model, **model_kwargs):
        # we only train velocity, hence only need to compute score from velocity
        score_out = self.schedule.get_score_from_velocity(model(x, t, **model_kwargs), x, t)
        return score_out

    def check_interval(self, diffusion_form="sigma", reverse=False, last_step_size=0.04):
        t0 = 0
        t1 = 1
        eps = self.sample_eps
        if (isinstance(self.schedule, GVPSchedule) or isinstance(self.schedule, LinearSchedule)):
            # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if diffusion_form == "SBDM" else 0
            t1 = 1 - eps if last_step_size == 0 else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def __get_sde_diffusion_and_drift(
        self,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.schedule.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def last_step(
        self,
        x,
        t,
        model,
        sde_drift,
        last_step,
        last_step_size,
        **model_kwargs
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            return x
        
        elif last_step == "Mean":
            return x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        
        elif last_step == "Tweedie":
            alpha = self.schedule.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.schedule.compute_sigma_t
            # return x / alpha(t)[0] + (sigma(t)[0] ** 2) / alpha(t)[0] * self.score(x, t, model, **model_kwargs)
            raise NotImplementedError("Tweedie last step seems weird (alpha(t) is indexed twice?!?)")
        
        elif last_step == "Euler":
            return x + self.drift(x, t, model, **model_kwargs) * last_step_size
        
        else:
            raise NotImplementedError(f"Last step '{last_step}' not implemented.")
    
    def sample(
        self,
        init,
        model,
        sampling_method="euler",
        diffusion_form="sigma",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
        progress=True,
        return_intermediates=False,
        cfg_scale=1.0,
        uc_cond=None,
        cond_key="y",
        **model_kwargs
    ):
        """
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.check_interval(diffusion_form=diffusion_form, reverse=False, last_step_size=last_step_size)
        ts = torch.linspace(t0, t1, num_steps).to(init.device)
        dt = ts[1] - ts[0]

        # enable classifier-free guidance 
        model_forward_fn = partial(forward_with_cfg, model=model, cfg_scale=cfg_scale, uc_cond=uc_cond, cond_key=cond_key)

        """ forward loop of sde """
        sampler = StepSDE(dt=dt, drift=sde_drift, diffusion=sde_diffusion, sampler_type=sampling_method)
        
        # sample
        x = init
        mean_x = init
        xs = []
        for ti in tqdm(ts[:-1], disable=not progress, desc="SDE sampling", total=num_steps, initial=1):
            with torch.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model_forward_fn, **model_kwargs)
                xs.append(x)
        
        # make last step
        t_last = torch.ones(x.size(0), device=x.device) * t1
        x = self.last_step(
            x=xs[-1], t=t_last,
            model=model_forward_fn,
            sde_drift=sde_drift,
            last_step=last_step,
            last_step_size=last_step_size,
            **model_kwargs
        )
        xs.append(x)

        assert len(xs) == num_steps, "Samples does not match the number of steps"

        if return_intermediates:
            return xs
        return xs[-1]
    

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

