"""Bundle finished runs and move them to remote storage to keep the local machine clean.

A run is a directory below ``--logs`` that contains ``config.yaml`` (written by train.py). Each run is
packed into ``<run>.tar.gz`` with a ``MANIFEST.json`` (sha256 of every file), uploaded, and - with
``--prune`` - deleted locally. A small ``ARCHIVED.json`` stub records where the run went.
Checkpoints are left out unless ``--include-checkpoints`` is given.

    python scripts/archive_runs.py --dest /mnt/nas/ngslfm-runs                       # local / mounted disk
    python scripts/archive_runs.py --dest user@storage:/archive/ngslfm/runs --prune  # SSH (rsync)
    python scripts/archive_runs.py --dest s3remote:ngslfm-runs --transport rclone    # S3/GCS/... (rclone)
"""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

STUB = "ARCHIVED.json"
CHECKPOINT_DIR = "checkpoints"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_runs(logs: Path, min_age_minutes: float) -> list[Path]:
    """Run directories that are not archived yet and have not been written to recently."""
    cutoff = time.time() - 60 * min_age_minutes
    runs = []
    for config in sorted(logs.rglob("config.yaml")):
        run = config.parent
        if (run / STUB).exists():
            continue
        if max(p.stat().st_mtime for p in run.rglob("*") if p.is_file()) <= cutoff:
            runs.append(run)
    return runs


def run_files(run: Path, include_checkpoints: bool) -> list[Path]:
    return sorted(
        p for p in run.rglob("*")
        if p.is_file() and (include_checkpoints or CHECKPOINT_DIR not in p.relative_to(run).parts)
    )


def bundle(run: Path, logs: Path, files: list[Path], out_dir: Path) -> Path:
    name = "__".join(run.relative_to(logs).parts)
    manifest = {
        "run": str(run.relative_to(logs)),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "files": {str(p.relative_to(run)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }
    archive = out_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{name}/{path.relative_to(run)}")
        manifest_path = out_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        tar.add(manifest_path, arcname=f"{name}/MANIFEST.json")
    return archive


def upload(archive: Path, dest: str, transport: str) -> None:
    if transport == "rsync":
        command = ["rsync", "--checksum", "--partial", str(archive), dest.rstrip("/") + "/"]
    else:
        command = ["rclone", "copy", "--checksum", str(archive), dest]
    subprocess.run(command, check=True)


def prune(run: Path, files: list[Path]) -> None:
    for path in files:
        path.unlink()
    for directory in sorted((d for d in run.rglob("*") if d.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


def archive_runs(logs: Path, dest: str, transport: str = "rsync", include_checkpoints: bool = False,
                 prune_local: bool = False, min_age_minutes: float = 30, dry_run: bool = False) -> list[Path]:
    if transport == "rsync" and ":" not in dest:
        Path(dest).mkdir(parents=True, exist_ok=True)
    archived = []
    for run in find_runs(logs, min_age_minutes):
        files = run_files(run, include_checkpoints)
        print(f"[archive] {run} ({len(files)} files) -> {dest}")
        if dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            archive = bundle(run, logs, files, Path(tmp))
            upload(archive, dest, transport)
            stub = {"destination": f"{dest.rstrip('/')}/{archive.name}", "sha256": sha256(archive),
                    "archived": datetime.datetime.now().isoformat(timespec="seconds"), "pruned": prune_local}
        if prune_local:
            prune(run, files)
        (run / STUB).write_text(json.dumps(stub, indent=2))
        archived.append(run)
    return archived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=Path("logs_dir"))
    parser.add_argument("--dest", required=True, help="local dir, user@host:/path (rsync) or remote:path (rclone)")
    parser.add_argument("--transport", choices=["rsync", "rclone"], default="rsync")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--prune", action="store_true", help="delete archived files locally after the upload")
    parser.add_argument("--min-age-minutes", type=float, default=30, help="skip runs written to more recently")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if shutil.which(args.transport) is None:
        parser.error(f"'{args.transport}' is not installed")
    archive_runs(args.logs, args.dest, args.transport, args.include_checkpoints, args.prune,
                 args.min_age_minutes, args.dry_run)
