"""Download the public pre-trained weights the framework builds on (skips files that already exist).

    python scripts/download_weights.py [--dir checkpoints]        (make weights)

* ``sd_ae.ckpt``            Stable Diffusion KL-autoencoder (Stage 1), state dict for jutils.nn.AutoencoderKL
* ``SiT-XL-2-256x256.pt``   class-conditional SiT-XL/2 on ImageNet 256 (Ma et al., 2024)
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

WEIGHTS = {
    "sd_ae.ckpt": "https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt"
                  "?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=1",
    "SiT-XL-2-256x256.pt": "https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt"
                           "?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0",
}


def download(url: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(target)          # only complete files get the final name


def main(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in WEIGHTS.items():
        target = directory / name
        if target.exists():
            print(f"[skip] {target} exists")
            continue
        print(f"[download] {name} -> {target}")
        download(url, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path("checkpoints"))
    main(parser.parse_args().dir)
