import os
import sys
import signal
import datetime

# ----
from omegaconf import OmegaConf, DictConfig
from torch.profiler import ProfilerActivity

# ---
import hydra
import torch

# lightning stuff
from lightning import Trainer
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.profilers import PyTorchProfiler

# ddp stuff
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from jutils import instantiate_from_config
from jutils import count_parameters, exists

# ldm stuff
from ldm.helpers import load_model_weights


torch.set_float32_matmul_precision('high')


def check_config(cfg):
    if cfg.get("auto_requeue", False):
        raise NotImplementedError("Auto-requeuing not working yet!")
    if exists(cfg.get("resume_checkpoint", None)) and exists(cfg.get("load_weights", None)):
        raise ValueError("Can't resume checkpoint and load weights at the same time.")
    if "experiment" in cfg:
        raise ValueError("Experiment config not merged successfully!")
    if cfg.use_wandb and cfg.use_wandb_offline:
        raise ValueError("Decide either for Online or Offline wandb, not both.")
    
    

@hydra.main(config_path="configs", config_name="bvae_config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(2025)

    """ Check config """
    check_config(cfg)
    print(cfg)

    """" Setup Logging """
    # we store the experiment under: logs/<cfg.name>/<day>/<slurm-id OR timestamp>
    day = datetime.datetime.now().strftime("%Y-%m-%d")
    postfix = str(cfg.slurm_id) if exists(cfg.slurm_id) else datetime.datetime.now().strftime("T%H%M%S")
    exp_name = os.path.join(cfg.name, day, postfix)
    log_dir = os.path.join("logs_dir", exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")

    # setup loggers
    use_wandb_logging = cfg.use_wandb or cfg.use_wandb_offline
    if use_wandb_logging:
        mode = "offline" if cfg.use_wandb_offline else "online"
        online_logger = WandbLogger(
            dir=log_dir,
            save_dir=log_dir,
            name=exp_name,
            tags=[cfg.user, *cfg["tags"]],
            project=cfg.wandb_project,
            config=OmegaConf.to_object(cfg),
            mode=mode,
            group="DDP"
        )
    else:
        online_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
            version="",
            log_graph=False,
            default_hp_metric=False,
        )
    csv_logger = CSVLogger(
        log_dir,
        name="",
        version="",
        prefix="",
        flush_logs_every_n_steps=500
    )
    csv_logger.log_hyperparams(OmegaConf.to_container(cfg))
    logger = [online_logger, csv_logger]

    """ Setup dataloader """
    data = instantiate_from_config(cfg.data)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup(stage="fit")

    """ Setup model """
    module = instantiate_from_config(cfg.trainer_module)
    
    """ Setup callbacks """
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step{step:06d}",
        # from config
        **cfg.checkpoint_params
    )
    callbacks = [checkpoint_callback]

    # add tqdm progress bar callback
    if cfg.tqdm_refresh_rate != 1:
        from lightning.pytorch.callbacks import TQDMProgressBar
        tqdm_callback = TQDMProgressBar(refresh_rate=cfg.tqdm_refresh_rate)
        callbacks.append(tqdm_callback)

    # other callbacks from config
    for cb_cfg in cfg.callbacks:
        cb = instantiate_from_config(cb_cfg)
        callbacks.append(cb)

    """ Profiling """
    if cfg.profile:
        profiler = PyTorchProfiler(
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
            schedule=torch.profiler.schedule(wait=0, warmup=cfg.profiling.warmup, active=cfg.profiling.active, repeat=1)
        )
        print(f"[Profiling] Enabled after {cfg.profiling.warmup} steps.")
    else:
        profiler = None
        
        
    """ Setup trainer """
    if torch.cuda.is_available():
        print("Using GPU")
        gpu_kwargs = {'accelerator': 'gpu', 'strategy': 'ddp'}
        if cfg.devices > 0:
            gpu_kwargs["devices"] = cfg.devices
        else:       # determine automatically
            gpu_kwargs["devices"] = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
        gpu_kwargs["num_nodes"] = cfg.num_nodes
        if cfg.num_nodes >= 2:
            if cfg.deepspeed_stage > 0:
                gpu_kwargs["strategy"] = f'deepspeed_stage_{cfg.deepspeed_stage}'
            else:
                # multi-node hacks from
                # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html
                gpu_kwargs["strategy"] = DDPStrategy(
                    gradient_as_bucket_view=True,
                    ddp_comm_hook=default_hooks.fp16_compress_hook
                )
        if cfg.auto_requeue:
            gpu_kwargs["plugins"] = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGUSR1)]
        if cfg.p2p_disable:
            # multi-gpu hack for heidelberg servers
            os.environ["NCCL_P2P_DISABLE"] = "1"
    else:
        print("Using CPU")
        gpu_kwargs = {'accelerator': 'cpu'}

    # Monitor gpu stats
    device_stats = DeviceStatsMonitor(cpu_stats=True)
    callbacks.append(device_stats)   

    # Early stopping
    # early_stopping = EarlyStopping(monitor="val/loss", mode="min", patience=1000) 
    # callbacks.append(early_stopping)

    # Add custom trainer params
    trainer_params = OmegaConf.to_container(cfg.trainer_params)
    if cfg.profile: trainer_params.update({'max_steps': cfg.profiling.warmup + cfg.profiling.active})
    
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **gpu_kwargs,
        # from config
        **trainer_params
    )
    
    """ Setup signal handler """
    # hacky way to avoid define this in the trainer module
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

    # once the signal was sent, the stop_training flag tells
    # the pl module get ready for save checkpoint
    def signal_handler(sig, frame):
        print(f"Activate signal handler for signal {sig}")
        module.stop_training = True

    signal.signal(signal.SIGUSR1, signal_handler)

    """ Log some information """
    # compute global batchsize
    bs = cfg.data.params.batch_size
    bs = bs * gpu_kwargs["devices"]
    bs = bs * gpu_kwargs["num_nodes"]
    bs = bs * cfg.trainer_params["accumulate_grad_batches"]
    # log info
    some_info = {
        'Command': " ".join(["python"] + sys.argv),
        'Name': exp_name,
        'Log dir': log_dir,
        'Trainer Module': cfg.trainer_module["target"],
        'Params': count_parameters(module),
        'Data': cfg.data.get("name", "not set"),
        'Batchsize': cfg.data.params.batch_size,
        'Devices': gpu_kwargs["devices"],
        'Num nodes': gpu_kwargs["num_nodes"],
        'Gradient accum': cfg.trainer_params["accumulate_grad_batches"],
        'Global batchsize': bs,
        'LR': cfg.trainer_module.params.lr,
        'LR scheduler': cfg.lr_scheduler.get("name", "no name") if "lr_scheduler" in cfg else "None",
        'Resume ckpt': cfg.resume_checkpoint,
        'Load weights': cfg.load_weights,
        'Profiling': f"Step {cfg.profiling.warmup}" if cfg.profile else "None",
        'Precision': cfg.trainer_params.precision,
    }
    
    # Make sure we don't log multiple times
    if trainer.global_rank == 0:
        print("-" * 40)
        for k, v in gpu_kwargs.items():
            print(f"{k:<16}: {v}")
        print("-" * 40)
        for k, v in some_info.items():
            if use_wandb_logging:
                online_logger.experiment.summary[k] = v
            if isinstance(v, float):
                print(f"{k:<16}: {v:.5f}")
            elif isinstance(v, int):
                print(f"{k:<16}: {v:,}")
            elif isinstance(v, bool):
                print(f"{k:<16}: {'True' if v else 'False'}")
            else:
                print(f"{k:<16}: {v}")
        print("-" * 40)
        # log called command
        if use_wandb_logging:
            online_logger.experiment.summary["command"] = " ".join(["python"] + sys.argv)
        
        # save config file
        OmegaConf.save(cfg, f"{log_dir}/config.yaml")
        # add command to config
        with open(f"{log_dir}/config.yaml", "a") as f:
            f.write("\n# Command\n")
            f.write(" ".join(["#"] + sys.argv))

    """ Train """
    ckpt_path = cfg.resume_checkpoint if exists(cfg.resume_checkpoint) else None
    if exists(cfg.load_weights):
        module = load_model_weights(module, cfg.load_weights, strict=True)
    trainer.fit(module, train_dataloaders=data.train_dataloader(), 
                val_dataloaders=data.val_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()