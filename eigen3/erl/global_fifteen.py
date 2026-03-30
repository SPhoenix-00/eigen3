"""Global 15 — shared leaderboard of agents that pass the HoF gauntlet.

Project-wide store (local under ``EIGEN3_ARTIFACT_ROOT/global_15/`` and optional
GCS prefix ``{cloud_project}/global_15/``), separate from per-run HoF.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eigen3.agents.trading_agent import (
    TradingNetworkParams,
    params_for_flax_msgpack,
    trading_params_from_msgpack_bytes,
)
from eigen3.erl.cloud_sync import CloudSync

logger = logging.getLogger(__name__)

AGENT_FILE_SUFFIX = ".msgpack"
META_FILENAME = "global_fifteen.json"


@dataclass
class GlobalFifteenEntry:
    agent_id: int
    gauntlet_score: float
    val_bn_excess: float
    hold_bn_excess: float
    promoted_at_generation: int
    source_run_name: str
    source_hof_agent_id: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "GlobalFifteenEntry":
        return GlobalFifteenEntry(
            agent_id=int(data["agent_id"]),
            gauntlet_score=float(data["gauntlet_score"]),
            val_bn_excess=float(data["val_bn_excess"]),
            hold_bn_excess=float(data["hold_bn_excess"]),
            promoted_at_generation=int(data["promoted_at_generation"]),
            source_run_name=str(data.get("source_run_name", "")),
            source_hof_agent_id=int(data.get("source_hof_agent_id", -1)),
        )


def _save_params(params: Any, path: Path) -> None:
    from flax.serialization import to_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = to_bytes(params_for_flax_msgpack(params))
    path.write_bytes(raw)


def _load_params(path: Path, target: Any) -> Any:
    from flax.serialization import from_bytes

    raw = path.read_bytes()
    if isinstance(target, TradingNetworkParams):
        return trading_params_from_msgpack_bytes(raw, target)
    return from_bytes(target, raw)


def _safe_unlink(path: Path, max_retries: int = 5, base_delay: float = 0.1) -> bool:
    if not path.exists():
        return True
    for attempt in range(max_retries):
        try:
            path.unlink()
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
            else:
                logger.warning("Could not delete %s (file locked)", path.name)
                return False
    return False


class GlobalFifteen:
    """Top-N gauntlet survivors; ranked by ``gauntlet_score`` (replace minimum when full)."""

    def __init__(
        self,
        capacity: int = 15,
        checkpoint_dir: Optional[Path] = None,
        cloud_sync: Optional[CloudSync] = None,
        cloud_prefix: Optional[str] = None,
    ):
        self.capacity = capacity
        self.checkpoint_dir = checkpoint_dir
        self.cloud_sync = cloud_sync
        self.cloud_prefix = cloud_prefix or "eigen3/global_15"

        self.entries: List[GlobalFifteenEntry] = []

        if self.checkpoint_dir:
            self.g15_dir = self.checkpoint_dir
            self.g15_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.g15_dir = None

    def _next_id(self) -> int:
        if not self.entries:
            return 0
        return max(e.agent_id for e in self.entries) + 1

    def _agent_path(self, agent_id: int) -> Path:
        assert self.g15_dir is not None
        return self.g15_dir / f"g15_agent_{agent_id}{AGENT_FILE_SUFFIX}"

    def __len__(self) -> int:
        return len(self.entries)

    def snapshot_entries_sorted(self) -> List[GlobalFifteenEntry]:
        return sorted(self.entries, key=lambda e: e.gauntlet_score, reverse=True)

    def add_or_replace(
        self,
        params: Any,
        gauntlet_score: float,
        val_bn_excess: float,
        hold_bn_excess: float,
        promoted_at_generation: int,
        source_run_name: str,
        source_hof_agent_id: int,
    ) -> str:
        """Admit *params* if there is room or *gauntlet_score* beats the current minimum.

        Returns ``"admitted"``, ``"replaced_<score>"``, or ``"rejected_not_better"``.
        """
        if self.g15_dir is None:
            return "rejected_not_better"

        if len(self.entries) < self.capacity:
            new_id = self._next_id()
            entry = GlobalFifteenEntry(
                agent_id=new_id,
                gauntlet_score=gauntlet_score,
                val_bn_excess=val_bn_excess,
                hold_bn_excess=hold_bn_excess,
                promoted_at_generation=promoted_at_generation,
                source_run_name=source_run_name,
                source_hof_agent_id=source_hof_agent_id,
            )
            self.entries.append(entry)
            _save_params(params, self._agent_path(new_id))
            return "admitted"

        worst = min(self.entries, key=lambda e: e.gauntlet_score)
        if gauntlet_score <= worst.gauntlet_score:
            return "rejected_not_better"

        old_score = worst.gauntlet_score
        self.entries.remove(worst)
        _safe_unlink(self._agent_path(worst.agent_id))

        new_id = self._next_id()
        entry = GlobalFifteenEntry(
            agent_id=new_id,
            gauntlet_score=gauntlet_score,
            val_bn_excess=val_bn_excess,
            hold_bn_excess=hold_bn_excess,
            promoted_at_generation=promoted_at_generation,
            source_run_name=source_run_name,
            source_hof_agent_id=source_hof_agent_id,
        )
        self.entries.append(entry)
        _save_params(params, self._agent_path(new_id))
        return f"replaced_{old_score:.6f}"

    def load_params(self, agent_id: int, target: Any) -> Any:
        path = self._agent_path(agent_id)
        if not path.exists():
            raise FileNotFoundError(f"Global 15 agent file not found: {path}")
        return _load_params(path, target)

    def save(self) -> None:
        if self.g15_dir is None:
            return

        metadata = {
            "capacity": self.capacity,
            "entries": [e.to_dict() for e in self.snapshot_entries_sorted()],
        }
        meta_path = self.g15_dir / META_FILENAME
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.cloud_sync and self.cloud_sync.provider != "local":
            cloud_json = f"{self.cloud_prefix}/{META_FILENAME}"
            self.cloud_sync.upload_file(str(meta_path), cloud_json, background=True)

            for entry in self.entries:
                local_agent = self._agent_path(entry.agent_id)
                if local_agent.exists():
                    cloud_agent = (
                        f"{self.cloud_prefix}/g15_agent_{entry.agent_id}{AGENT_FILE_SUFFIX}"
                    )
                    self.cloud_sync.upload_file(str(local_agent), cloud_agent, background=True)

    def load(self) -> None:
        if self.g15_dir is None:
            return

        meta_path = self.g15_dir / META_FILENAME

        if (
            not meta_path.exists()
            and self.cloud_sync
            and self.cloud_sync.provider != "local"
        ):
            cloud_json = f"{self.cloud_prefix}/{META_FILENAME}"
            self.cloud_sync.download_file(cloud_json, str(meta_path))

        if not meta_path.exists():
            return

        with open(meta_path) as f:
            metadata = json.load(f)

        self.capacity = int(metadata.get("capacity", self.capacity))
        self.entries = [
            GlobalFifteenEntry.from_dict(e) for e in metadata.get("entries", [])
        ]

        if self.cloud_sync and self.cloud_sync.provider != "local":
            for entry in self.entries:
                local_agent = self._agent_path(entry.agent_id)
                if not local_agent.exists():
                    cloud_agent = (
                        f"{self.cloud_prefix}/g15_agent_{entry.agent_id}{AGENT_FILE_SUFFIX}"
                    )
                    self.cloud_sync.download_file(cloud_agent, str(local_agent), silent=True)

        logger.info(
            "Global 15 loaded: %d entries (best=%.4f, worst=%.4f)",
            len(self.entries),
            max((e.gauntlet_score for e in self.entries), default=0.0),
            min((e.gauntlet_score for e in self.entries), default=0.0),
        )

    def to_snapshot_dicts(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.snapshot_entries_sorted()]
