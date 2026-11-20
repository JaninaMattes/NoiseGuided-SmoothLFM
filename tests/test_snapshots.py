"""The resolved thesis configurations are frozen in tests/snapshots/.

Any change to a hyper-parameter, model size, dataset setting or tracking default of the thesis
setup makes this test fail with a diff. If the change is intended, accept it with
``make snapshots`` and commit the updated snapshot together with the change, so the
experimental setup never drifts unnoticed.
"""
import difflib
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from conftest import compose_config

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOTS = {               # snapshot file -> (root config, overrides)
    "train_beta_vae": ("train_beta_vae", []),
    "train_flow_decoder": ("train_flow_decoder", []),
    "build_dataset": ("build_dataset", []),
}


@pytest.mark.parametrize("name", SNAPSHOTS)
def test_resolved_config_matches_snapshot(name, request):
    config_name, overrides = SNAPSHOTS[name]
    current = OmegaConf.to_yaml(compose_config(config_name, overrides), resolve=True)
    path = SNAPSHOT_DIR / f"{name}.yaml"

    if request.config.getoption("--update-snapshots"):
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path.write_text(current)
        return
    assert path.exists(), f"missing snapshot {path}; create it with `make snapshots`"
    expected = path.read_text()
    diff = "".join(difflib.unified_diff(expected.splitlines(True), current.splitlines(True), str(path), "current"))
    assert not diff, f"Experimental setup changed (accept with `make snapshots` if intended):\n{diff}"
