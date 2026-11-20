# Archiving runs

`make archive-runs` packs each finished run in `logs_dir/` (configs, `metrics.csv`, logs and images) into one
`.tar.gz` with a SHA-256 manifest, copies it to `RUNS_REMOTE`, and leaves an `ARCHIVED.json` stub that records
where the run went.

```bash
export RUNS_REMOTE=user@storage-host:/archive/ngslfm/runs    # suggested default, see below
make archive-runs ARGS="--dry-run"                            # list what would be archived
make archive-runs ARGS="--prune"                              # upload, then delete the archived files locally
```

- **Finished runs only.** A run directory written to in the last 30 minutes is skipped (`--min-age-minutes`).
  Runs that already have a stub are never archived twice.
- **Checkpoints stay local** unless you pass `--include-checkpoints`. `--prune` only deletes files that were
  uploaded.
- **Verified copies.** `rsync`/`rclone` copy with `--checksum`. The stub stores the archive's SHA-256.

## Where to store runs

Suggested layout: one folder per project on storage that outlives your machine. The archive name encodes the run
(`beta_vae__0.5x-1.0x-0.1b__2025-07-01__T101500.tar.gz`), so you can keep everything in one flat folder.

| Storage | `RUNS_REMOTE` | Transport |
|---|---|---|
| Cluster / NAS over SSH (suggested) | `user@storage-host:/archive/ngslfm/runs` | `rsync` (default) |
| Mounted drive | `/Volumes/Archive/ngslfm/runs` | `rsync` (default) |
| S3, GCS, Azure, Google Drive, … | `myremote:ngslfm-runs` (set up with `rclone config`) | `ARGS="--transport rclone"` |

To restore a run: `tar -xzf <run>.tar.gz`. `MANIFEST.json` inside lists every file with its checksum.

MLflow keeps its own copy of metrics and images in `./mlruns`. To keep that off your machine too, use a shared
tracking server ([mlflow.md](mlflow.md)).
