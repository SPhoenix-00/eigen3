import random
from pathlib import Path

import eigen3.utils.run_naming as rn


def test_next_index_for_adjective_noun(tmp_path: Path):
    (tmp_path / "swift-river-1").mkdir()
    (tmp_path / "swift-river-3").mkdir()
    assert rn.next_index_for_adjective_noun(tmp_path, "swift", "river") == 4


def test_generate_wandb_style_run_name_format(tmp_path: Path):
    name = rn.generate_wandb_style_run_name(tmp_path)
    parts = name.split("-")
    assert len(parts) == 3
    assert parts[2].isdigit()


def test_generate_wandb_style_same_rng_two_empty_roots(tmp_path_factory):
    p1 = tmp_path_factory.mktemp("a")
    p2 = tmp_path_factory.mktemp("b")
    r = random.Random(12345)
    a = rn.generate_wandb_style_run_name(p1, rng=r)
    r = random.Random(12345)
    b = rn.generate_wandb_style_run_name(p2, rng=r)
    assert a == b
