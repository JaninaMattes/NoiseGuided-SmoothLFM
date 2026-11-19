"""Single training entry point for both Stage-2 models of the framework.

    python train.py                                   # β-VAE self-guidance encoder (configs/bvae_config.yaml)
    python train.py --config-name rf_config           # self-guided flow matching decoder (configs/rf_config.yaml)
    python train.py experiment=imnet256/bvae-skipvit-b-2 use_mlflow=True

Any config value can be overridden on the command line (Hydra syntax).
Runs are stored under ``logs_dir/<name>/<day>/<slurm-id | timestamp>``.
"""
import datetime
import os
import signal
import sys

import hydra
import torch
from jutils import count_parameters, exists, instantiate_from_config
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.profiler import ProfilerActivity

from ldm.helpers import load_model_weights
from ldm.tracking import build_loggers, log_run_info

torch.set_float32_matmul_precision("high")


def check_config(cfg: DictConfig) -> None:
    if cfg.get("auto_requeue", False):
        raise NotImplementedError("Auto-requeuing not working yet!")
    if exists(cfg.get("resume_checkpoint", None)) and exists(cfg.get("load_weights", None)):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")
    if "experiment" in cfg:
        raise ValueError("Experiment config not merged successfully!")
    if cfg.use_wandb and cfg.use_wandb_offline:
        raise ValueError("Decide either for Online or Offline wandb, not both.")


def build_profiler(cfg: DictConfig, log_dir: str):
    if not cfg.profile:
        return None
    print(f"[Profiling] Enabled after {cfg.profiling.warmup} steps.")
    return PyTorchProfiler(
        dirpath=log_dir,
        filename=cfg.profiling.filename,
        activities=[
            *((ProfilerActivity.CPU,) if cfg.profiling.cpu else ()),
            *((ProfilerActivity.CUDA,) if cfg.profiling.cuda else ()),
        ],
        record_shapes=cfg.profiling.record_shapes,
        profile_memory=cfg.profiling.profile_memory,
        with_flops=cfg.profiling.with_flops,
        with_stack=True,
        export_to_chrome=True,
        record_module_names=True,
        schedule=torch.profiler.schedule(wait=0, warmup=cfg.profiling.warmup, active=cfg.profiling.active, repeat=1),
    )


def device_kwargs(cfg: DictConfig) -> dict:
    """Accelerator, devices and distributed strategy for the Lightning Trainer."""
    if not torch.cuda.is_available():
        print("Using CPU")
        return {"accelerator": "cpu", "devices": 1, "num_nodes": 1}

    print("Using GPU")
    kwargs = {
        "accelerator": "gpu",
        "strategy": "ddp",
        "devices": cfg.devices if cfg.devices > 0 else torch.cuda.device_count(),
        "num_nodes": cfg.num_nodes,
    }
    if cfg.num_nodes >= 2:
        if cfg.deepspeed_stage > 0:
            kwargs["strategy"] = f"deepspeed_stage_{cfg.deepspeed_stage}"
        else:
            # multi-node hacks from https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
            kwargs["strategy"] = DDPStrategy(gradient_as_bucket_view=True, ddp_comm_hook=default_hooks.fp16_compress_hook)
    if cfg.auto_requeue:
        kwargs["plugins"] = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]
    if cfg.p2p_disable:
        os.environ["NCCL_P2P_DISABLE"] = "1"  # multi-gpu hack for heidelberg servers
    return kwargs


@hydra.main(config_path="configs", config_name="bvae_config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    check_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    """ Run directory: logs_dir/<cfg.name>/<day>/<slurm-id OR timestamp> """
    now = datetime.datetime.now()
    postfix = str(cfg.slurm_id) if exists(cfg.slurm_id) else now.strftime("T%H%M%S")
    exp_name = os.path.join(cfg.name, now.strftime("%Y-%m-%d"), postfix)
    log_dir = os.path.join("logs_dir", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    loggers = build_loggers(cfg, log_dir, run_name=exp_name)

    """ Data and model """
    data = instantiate_from_config(cfg.data)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup(stage="fit")
    module = instantiate_from_config(cfg.trainer_module)

    """ Callbacks """
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_dir, filename="step{step:06d}", **cfg.checkpoint_params),
        DeviceStatsMonitor(cpu_stats=True),
    ]
    if cfg.tqdm_refresh_rate != 1:
        callbacks.append(TQDMProgressBar(refresh_rate=cfg.tqdm_refresh_rate))
    callbacks.extend(instantiate_from_config(cb_cfg) for cb_cfg in cfg.callbacks)

    """ Trainer """
    gpu_kwargs = device_kwargs(cfg)
    trainer_params = OmegaConf.to_container(cfg.trainer_params)
    if cfg.profile:
        trainer_params["max_steps"] = cfg.profiling.warmup + cfg.profiling.active
    trainer = Trainer(logger=loggers, callbacks=callbacks, profiler=build_profiler(cfg, log_dir), **gpu_kwargs, **trainer_params)

    """ SLURM signal handling: save a checkpoint before the job is pre-empted """
    def stop_training_method():
        module.stop_training = False
        print("-" * 40)
        print("Try to save checkpoint to {}".format(ckpt_dir))
        module.trainer.save_checkpoint(os.path.join(ckpt_dir, "interrupted.ckpt"))
        module.trainer.should_stop = True
        module.trainer.limit_val_batches = 0
        print("Saved checkpoint.")
        print("-" * 40)

    module.stop_training_method = stop_training_method

    def signal_handler(sig, frame):
        print(f"Activate signal handler for signal {sig}")
        module.stop_training = True

    signal.signal(signal.SIGUSR1, signal_handler)

    """ Log run information """
    command = " ".join(["python"] + sys.argv)
    accumulate = cfg.trainer_params["accumulate_grad_batches"]
    batch_size = cfg.data.params.batch_size
    run_info = {
        "Command": command,
        "Name": exp_name,
        "Log dir": log_dir,
        "Trainer Module": cfg.trainer_module["target"],
        "Params": count_parameters(module),
        "Data": cfg.data.get("name", "not set"),
        "Batchsize": batch_size,
        "Devices": gpu_kwargs["devices"],
        "Num nodes": gpu_kwargs["num_nodes"],
        "Gradient accum": accumulate,
        "Global batchsize": batch_size * gpu_kwargs["devices"] * gpu_kwargs["num_nodes"] * accumulate,
        "LR": cfg.trainer_module.params.lr,
        "LR scheduler": cfg.lr_scheduler.get("name", "no name") if exists(cfg.get("lr_scheduler")) else "None",
        "Resume ckpt": cfg.resume_checkpoint,
        "Load weights": cfg.load_weights,
        "Profiling": f"Step {cfg.profiling.warmup}" if cfg.profile else "None",
        "Precision": cfg.trainer_params.precision,
        "Seed": cfg.seed,
    }
    if trainer.global_rank == 0:
        print("-" * 40)
        for k, v in {**gpu_kwargs, **run_info}.items():
            print(f"{k:<16}: {v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"{k:<16}: {v}")
        print("-" * 40)
        os.makedirs(log_dir, exist_ok=True)
        config_path = os.path.join(log_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)
        with open(config_path, "a") as f:
            f.write("\n# Command\n# " + command + "\n")
        log_run_info(loggers, run_info, config_path)

    """ Train """
    ckpt_path = cfg.resume_checkpoint if exists(cfg.resume_checkpoint) else None
    if exists(cfg.load_weights):
        module = load_model_weights(module, cfg.load_weights, strict=True)
    trainer.fit(module, train_dataloaders=data.train_dataloader(),
                val_dataloaders=data.val_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
