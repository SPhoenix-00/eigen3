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


def test_resolved_checkpoint_root_env_checkpoint_root(tmp_path, monkeypatch):
    ck = tmp_path / "my_ckpts"
    ck.mkdir()
    monkeypatch.setenv("EIGEN3_CHECKPOINT_ROOT", str(ck))
    monkeypatch.delenv("EIGEN3_ARTIFACT_ROOT", raising=False)
    assert rn.resolved_checkpoint_root(tmp_path / "ignored") == ck.resolve()


def test_resolved_checkpoint_root_artifact_root(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("EIGEN3_CHECKPOINT_ROOT", raising=False)
    monkeypatch.setenv("EIGEN3_ARTIFACT_ROOT", str(repo))
    assert rn.resolved_checkpoint_root(repo) == (repo / "checkpoints").resolve()


def test_suffix_increments_after_run_dir_created(tmp_path):
    r = random.Random(42)
    first = rn.generate_wandb_style_run_name(tmp_path, rng=r)
    (tmp_path / first).mkdir(parents=True)
    r = random.Random(42)
    second = rn.generate_wandb_style_run_name(tmp_path, rng=r)
    assert first != second
    assert second.split("-")[-1] == "2"


def test_generate_wandb_style_same_rng_two_empty_roots(tmp_path_factory):
    p1 = tmp_path_factory.mktemp("a")
    p2 = tmp_path_factory.mktemp("b")
    r = random.Random(12345)
    a = rn.generate_wandb_style_run_name(p1, rng=r)
    r = random.Random(12345)
    b = rn.generate_wandb_style_run_name(p2, rng=r)
    assert a == b
