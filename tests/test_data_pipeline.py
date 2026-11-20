"""Dataset generation: forward ODE, HDF5 writer/reader contract, image sources and the full build."""
import h5py
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from conftest import REPO, compose_config
from dataprocessing.forward_ode import GuidanceFreeEncoder
from dataprocessing.hdf5_writer import HDF5DatasetWriter
from dataprocessing.sources import ClassFolderImages
from ldm.dataloader.dataloader.hdf5_dataloader import HDF5DataModule, latent_key
from ldm.testing import PoolingAutoencoder

TIMESTEPS = [1.0, 0.5, 0.0]


class ConstantVelocity(nn.Module):
    """v(x_t, t, y) = c: the Euler path is exact, x_t = x_1 - (1 - t) c."""

    def __init__(self, c: float):
        super().__init__()
        self.c = c
        self.calls = []

    def forward(self, x, t, y, cfg_scale, uc_cond):
        self.calls.append(dict(t=t.clone(), y=y.clone(), cfg_scale=cfg_scale, uc_cond=uc_cond.clone()))
        return torch.full_like(x, self.c)


def images(n: int) -> torch.Tensor:
    return torch.randint(0, 256, (n, 3, 256, 256), dtype=torch.uint8)


def test_forward_ode_stores_the_state_at_each_timestep():
    flow, first_stage = ConstantVelocity(c=2.0), PoolingAutoencoder()
    encoder = GuidanceFreeEncoder(first_stage, flow, TIMESTEPS, num_steps=10, cfg_scale=1.0, num_classes=1000)
    batch, labels = images(2), torch.tensor([3, 7])

    states = encoder(batch, labels)

    x1 = first_stage.encode(batch.float() / 127.5 - 1.0)
    assert set(states) == set(TIMESTEPS)
    for t, xt in states.items():
        torch.testing.assert_close(xt, x1 - (1.0 - t) * 2.0, atol=1e-5, rtol=0)
    assert len(flow.calls) == 10                                    # integrates from t=1 down to t=0
    assert flow.calls[0]["t"].tolist() == [1.0, 1.0]
    assert all(c["cfg_scale"] == 1.0 for c in flow.calls)           # guidance-free
    assert flow.calls[0]["y"].tolist() == [3, 7]
    assert flow.calls[0]["uc_cond"].tolist() == [1000, 1000]        # null label


def test_forward_ode_stops_at_the_smallest_requested_timestep():
    flow = ConstantVelocity(c=0.0)
    GuidanceFreeEncoder(PoolingAutoencoder(), flow, [1.0, 0.7], num_steps=10)(images(1), torch.tensor([0]))
    assert len(flow.calls) == 3


def test_forward_ode_rejects_timesteps_off_the_euler_grid():
    with pytest.raises(ValueError, match="not on the grid"):
        GuidanceFreeEncoder(PoolingAutoencoder(), ConstantVelocity(0.0), [0.55], num_steps=10)


def write_samples(path, n: int, settings=None):
    rng = np.random.default_rng(0)
    data = dict(
        images=rng.integers(0, 256, (n, 3, 256, 256), dtype=np.uint8),
        labels=np.arange(n) % 2,
        latents={t: rng.standard_normal((n, 4, 32, 32), dtype=np.float32) for t in TIMESTEPS},
    )
    with HDF5DatasetWriter(path, settings or {"ode_steps": 10}, TIMESTEPS, compression=None) as writer:
        writer.append("train", data["images"], data["labels"], data["latents"])
    return data


def test_writer_output_is_what_the_training_dataloader_reads(tmp_path):
    path = tmp_path / "data.hdf5"
    data = write_samples(path, n=4)

    module = HDF5DataModule(str(path), source_timestep=0.5, target_timestep=1.0, train_group="train",
                            val_group=None, batch_size=4, num_workers=0, balance_classes=False, p=0.0)
    module.setup()
    sample = module.datasets["train"][1]

    assert set(sample) == {"image", "label", latent_key(0.5), latent_key(1.0)}
    torch.testing.assert_close(sample["image"], torch.from_numpy(data["images"][1]).float() / 127.5 - 1.0)
    assert sample["label"].item() == 1
    for t in (0.5, 1.0):
        np.testing.assert_array_equal(sample[latent_key(t)].numpy(), data["latents"][t][1])


def test_writer_repairs_a_partially_written_batch(tmp_path):
    path = tmp_path / "data.hdf5"
    write_samples(path, n=4)
    with h5py.File(path, "a") as f:                                 # crash after writing images only
        f["train/images"].resize(6, axis=0)

    with HDF5DatasetWriter(path, {"ode_steps": 10}, TIMESTEPS) as writer:
        assert writer.num_samples("train") == 4
    with h5py.File(path, "r") as f:
        assert {len(d) for d in f["train"].values()} == {4}


def test_writer_refuses_to_mix_generation_settings(tmp_path):
    path = tmp_path / "data.hdf5"
    write_samples(path, n=2)
    with pytest.raises(ValueError, match="ode_steps"):
        with HDF5DatasetWriter(path, {"ode_steps": 100}, TIMESTEPS):
            pass


def make_class_folders(root, layout: dict[str, int]):
    for folder, count in layout.items():
        (root / folder).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (300, 260), color=(i, 0, 0)).save(root / folder / f"img{i}.png")


def test_class_folder_images_are_ordered_filtered_and_cropped(tmp_path):
    make_class_folders(tmp_path, {"class_7": 3, "class_2": 2, "class_5": 1})

    dataset = ClassFolderImages(str(tmp_path), class_labels=[2, 7], max_per_class=2, image_size=256)

    assert [(p.parent.name, p.name, y) for p, y in dataset.samples] == [
        ("class_2", "img0.png", 2), ("class_2", "img1.png", 2), ("class_7", "img0.png", 7), ("class_7", "img1.png", 7),
    ]
    image, label = dataset[0]
    assert image.shape == (3, 256, 256) and image.dtype == torch.uint8 and label == 2


def test_class_folder_images_reject_missing_classes_and_unknown_layouts(tmp_path):
    make_class_folders(tmp_path / "ok", {"class_0": 1})
    with pytest.raises(ValueError, match=r"requested classes \[3\]"):
        ClassFolderImages(str(tmp_path / "ok"), class_labels=[0, 3])

    make_class_folders(tmp_path / "bad", {"n01440764": 1, "n01443537": 1})
    with pytest.raises(ValueError, match="synset folders"):
        ClassFolderImages(str(tmp_path / "bad"))


def test_built_dataset_has_every_split_and_timestep(smoke_dataset):
    cfg = compose_config("build_dataset", ["experiment=smoke/dataset"])
    with h5py.File(smoke_dataset, "r") as f:
        assert set(f) == {"train", "validation"}
        for split, size in [("train", 4), ("validation", 2)]:   # 2 classes x max_per_class
            expected = {"images", "labels", *(latent_key(t) for t in cfg.timesteps)}
            assert set(f[split]) == expected
            assert {len(d) for d in f[split].values()} == {size}
            assert sorted(f[split]["labels"][:, 0]) == [0] * (size // 2) + [1] * (size // 2)
        image = torch.from_numpy(f["train/images"][:1]).float() / 127.5 - 1.0     # x_1 = E(x)
        torch.testing.assert_close(torch.from_numpy(f["train"][latent_key(1.0)][:1]), PoolingAutoencoder().encode(image))
        assert "ode_steps" in f.attrs and "provenance.git_commit" in f.attrs


def test_dataset_build_resumes_without_duplicating_samples(smoke_dataset, tmp_path):
    from dataprocessing.build import build_dataset

    copy = tmp_path / "copy.hdf5"
    copy.write_bytes(smoke_dataset.read_bytes())
    build_dataset(compose_config("build_dataset", ["experiment=smoke/dataset", f"output={copy}",
                                                   f"paths.data={REPO / 'data'}"]))
    with h5py.File(copy, "r") as f:
        assert len(f["train/images"]) == 4
