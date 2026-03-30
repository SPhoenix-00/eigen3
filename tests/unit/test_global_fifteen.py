"""Global 15 admission / replace-min."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from eigen3.erl.global_fifteen import GlobalFifteen


@pytest.fixture
def g15(tmp_path: Path) -> GlobalFifteen:
    g = GlobalFifteen(capacity=3, checkpoint_dir=tmp_path, cloud_sync=None)
    return g


def _params():
    return {"w": jnp.array([1.0], dtype=jnp.float32)}


def test_add_fills_until_capacity(g15: GlobalFifteen) -> None:
    p = _params()
    assert g15.add_or_replace(p, 10.0, 5.0, 5.0, 1, "run_a", 0) == "admitted"
    assert g15.add_or_replace(p, 20.0, 10.0, 10.0, 1, "run_a", 1) == "admitted"
    assert g15.add_or_replace(p, 30.0, 15.0, 15.0, 1, "run_a", 2) == "admitted"
    assert len(g15) == 3


def test_replace_only_if_strictly_better(g15: GlobalFifteen) -> None:
    p = _params()
    for s in (30.0, 20.0, 10.0):
        g15.add_or_replace(p, s, s / 2, s / 2, 1, "r", int(s))
    assert len(g15) == 3
    assert g15.add_or_replace(p, 5.0, 2.0, 3.0, 2, "r", 99) == "rejected_not_better"
    act = g15.add_or_replace(p, 25.0, 12.0, 13.0, 2, "r", 100)
    assert act.startswith("replaced_")
    scores = sorted(e.gauntlet_score for e in g15.entries)
    assert scores == [20.0, 25.0, 30.0]


def test_snapshot_sorted_desc(g15: GlobalFifteen) -> None:
    p = _params()
    g15.add_or_replace(p, 1.0, 1.0, 0.0, 1, "r", 1)
    g15.add_or_replace(p, 9.0, 5.0, 4.0, 1, "r", 2)
    snap = g15.snapshot_entries_sorted()
    assert [e.gauntlet_score for e in snap] == [9.0, 1.0]
