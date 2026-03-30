"""Hall of Fame purge API for gauntlet."""

from pathlib import Path

from eigen3.erl.hall_of_fame import HallOfFame, HallOfFameEntry


def test_remove_entries_by_agent_ids_deletes_files(tmp_path: Path) -> None:
    hof = HallOfFame(capacity=10, checkpoint_dir=tmp_path)
    hof.entries = [
        HallOfFameEntry(agent_id=0, validation_score=1.0, generation=1),
        HallOfFameEntry(agent_id=1, validation_score=2.0, generation=1),
    ]
    assert hof.hof_dir is not None
    for e in hof.entries:
        (hof.hof_dir / f"hof_agent_{e.agent_id}.msgpack").write_bytes(b"dummy")

    removed = hof.remove_entries_by_agent_ids({0})
    assert removed == [0]
    assert len(hof.entries) == 1
    assert hof.entries[0].agent_id == 1
    assert not (hof.hof_dir / "hof_agent_0.msgpack").exists()
    assert (hof.hof_dir / "hof_agent_1.msgpack").exists()


def test_remove_entries_unknown_ids_no_op() -> None:
    hof = HallOfFame(capacity=10, checkpoint_dir=None)
    hof.entries = [HallOfFameEntry(agent_id=5, validation_score=1.0, generation=1)]
    assert hof.remove_entries_by_agent_ids({99}) == []
