"""Loggers are built from the config and images reach every image-capable backend."""
import numpy as np
import pytest
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from omegaconf import OmegaConf

from ldm.tracking import build_loggers, log_image, log_run_info


def make_cfg(tmp_path, use_mlflow: bool):
    return OmegaConf.create({
        "user": "tester",
        "tags": ["unit-test"],
        "use_wandb": False,
        "use_wandb_offline": False,
        "wandb_project": "test-project",
        "use_mlflow": use_mlflow,
        "mlflow_tracking_uri": f"file:{tmp_path / 'mlruns'}",
        "mlflow_experiment": "test-project",
    })


@pytest.fixture
def image():
    return np.random.randint(0, 255, size=(32, 64, 3), dtype=np.uint8)


def test_mlflow_run_receives_image_and_config(tmp_path, image):
    loggers = build_loggers(make_cfg(tmp_path, use_mlflow=True), str(tmp_path / "run"), run_name="test-run")
    assert [type(lg) for lg in loggers] == [MLFlowLogger, CSVLogger]

    config_path = tmp_path / "config.yaml"
    config_path.write_text("lr: 1e-4\n")
    log_run_info(loggers, {"Command": "python train.py"}, str(config_path))
    log_image(loggers, "val/samples", image, step=3)

    mlflow_logger = loggers[0]
    client = mlflow_logger.experiment
    artifacts = [a.path for a in client.list_artifacts(mlflow_logger.run_id)]
    assert "config.yaml" in artifacts
    assert "images" in artifacts
    assert client.get_run(mlflow_logger.run_id).data.tags["info.Command"] == "python train.py"


def test_tensorboard_fallback_when_no_online_tracker(tmp_path, image):
    run_dir = tmp_path / "run"
    loggers = build_loggers(make_cfg(tmp_path, use_mlflow=False), str(run_dir), run_name="test-run")
    assert [type(lg) for lg in loggers] == [TensorBoardLogger, CSVLogger]

    log_image(loggers, "val/samples", image, step=0)
    loggers[0].experiment.flush()
    assert list(run_dir.glob("events.out.tfevents.*"))
