"""Hall of Fame — persistent archive of the best agents across generations.

Port of Eigen2's ``erl/hall_of_fame.py`` to JAX/Flax.  Key differences:

* Agent weights are serialised with ``flax.serialization`` (msgpack), not
  PyTorch ``.pth``.
* ``CloudSync`` handles GCS persistence so the HoF survives across runs and
  machines.
* No PyTorch / CUDA dependency.

The two-phase admission logic (fill → cascading swap) and EMA score erosion
are preserved verbatim from Eigen2.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eigen3.agents.trading_agent import (
    TradingNetworkParams,
    params_for_flax_msgpack,
    trading_params_from_msgpack_bytes,
)
from eigen3.erl.cloud_sync import CloudSync

logger = logging.getLogger(__name__)

AGENT_FILE_SUFFIX = ".msgpack"


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class HallOfFameEntry:
    """Metadata for one HoF slot (agent weights stored separately on disk)."""

    agent_id: int
    validation_score: float
    generation: int
    # Validation ROI (%): 100 * sum(episode net $ PnL) / max(episode peak capital employed)
    roi: float = 0.0
    expectancy: float = 0.0
    train_fitness: float = 0.0
    quality_count: int = 0
    total_trades: int = 0
    val_fitness: float = 0.0
    base_combined_fitness: float = 0.0
    # Mean eval episode bonus vs equal-weight buy-and-hold (scaled $, same as env)
    benchmark_excess_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "HallOfFameEntry":
        return HallOfFameEntry(
            agent_id=data["agent_id"],
            validation_score=data["validation_score"],
            generation=data["generation"],
            roi=data.get("roi", 0.0),
            expectancy=data.get("expectancy", 0.0),
            train_fitness=data.get("train_fitness", 0.0),
            quality_count=data.get("quality_count", 0),
            total_trades=data.get("total_trades", 0),
            val_fitness=data.get("val_fitness", 0.0),
            base_combined_fitness=data.get("base_combined_fitness", 0.0),
            benchmark_excess_usd=data.get("benchmark_excess_usd", 0.0),
        )


# ---------------------------------------------------------------------------
# Flax param helpers
# ---------------------------------------------------------------------------


def _save_params(params: Any, path: Path) -> None:
    """Serialise a Flax/JAX param pytree to *path* (msgpack)."""
    from flax.serialization import to_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = to_bytes(params_for_flax_msgpack(params))
    path.write_bytes(raw)


def _load_params(path: Path, target: Any) -> Any:
    """Deserialise params from *path* into *target* pytree structure."""
    from flax.serialization import from_bytes

    raw = path.read_bytes()
    if isinstance(target, TradingNetworkParams):
        return trading_params_from_msgpack_bytes(raw, target)
    return from_bytes(target, raw)


# ---------------------------------------------------------------------------
# Safe delete (Windows file-lock retry, same as Eigen2)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hall of Fame
# ---------------------------------------------------------------------------


class HallOfFame:
    """Persistent archive of the best agents by validation score.

    Mirrors Eigen2 semantics:

    * Two-phase admission (fill, then cascading swap).
    * EMA-based score erosion.
    * Saves / loads metadata as JSON, agent weights as msgpack.
    * Optionally syncs to a GCS bucket via ``CloudSync``.

    Args:
        capacity: Maximum number of agents to store.
        checkpoint_dir: Local directory for HoF data and agent weights.
        cloud_sync: Optional ``CloudSync`` for GCS persistence.
        cloud_prefix: Cloud path prefix (e.g. ``"eigen3/hall_of_fame"``).
    """

    def __init__(
        self,
        capacity: int = 10,
        checkpoint_dir: Optional[Path] = None,
        cloud_sync: Optional[CloudSync] = None,
        cloud_prefix: Optional[str] = None,
    ):
        self.capacity = capacity
        self.checkpoint_dir = checkpoint_dir
        self.cloud_sync = cloud_sync
        self.cloud_prefix = cloud_prefix or "eigen3/hall_of_fame"

        self.entries: List[HallOfFameEntry] = []

        if self.checkpoint_dir:
            self.hof_dir = self.checkpoint_dir / "hall_of_fame"
            self.hof_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.hof_dir = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity

    def get_worst_score(self) -> float:
        if not self.entries:
            return float("-inf")
        return min(e.validation_score for e in self.entries)

    def should_admit(self, validation_score: float) -> bool:
        if not self.is_full():
            return True
        return validation_score > self.get_worst_score()

    def get_median_roi(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.median([e.roi for e in self.entries]))

    def get_stats(self) -> Dict[str, Any]:
        if not self.entries:
            return {
                "size": 0,
                "best_score": 0.0,
                "worst_score": 0.0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "oldest_generation": 0,
                "newest_generation": 0,
                "median_roi": 0.0,
                "best_bh_excess": 0.0,
                "worst_bh_excess": 0.0,
            }
        scores = [e.validation_score for e in self.entries]
        gens = [e.generation for e in self.entries]
        best_e = max(self.entries, key=lambda e: e.validation_score)
        worst_e = min(self.entries, key=lambda e: e.validation_score)
        return {
            "size": len(self.entries),
            "best_score": float(max(scores)),
            "worst_score": float(min(scores)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "oldest_generation": min(gens),
            "newest_generation": max(gens),
            "median_roi": self.get_median_roi(),
            "best_bh_excess": float(best_e.benchmark_excess_usd),
            "worst_bh_excess": float(worst_e.benchmark_excess_usd),
        }

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        if not self.entries:
            return 0
        return max(e.agent_id for e in self.entries) + 1

    def _agent_path(self, agent_id: int) -> Path:
        assert self.hof_dir is not None
        return self.hof_dir / f"hof_agent_{agent_id}{AGENT_FILE_SUFFIX}"

    def add(
        self,
        params: Any,
        validation_score: float,
        generation: int,
        *,
        roi: float = 0.0,
        expectancy: float = 0.0,
        train_fitness: float = 0.0,
        quality_count: int = 0,
        total_trades: int = 0,
        val_fitness: float = 0.0,
        base_combined_fitness: float = 0.0,
        benchmark_excess_usd: float = 0.0,
    ) -> bool:
        """Attempt to add *params* (Flax pytree) to the HoF.

        Returns True if the agent was admitted.
        """
        if not self.should_admit(validation_score):
            return False

        if self.is_full():
            worst = min(self.entries, key=lambda e: e.validation_score)
            self.entries.remove(worst)
            if self.hof_dir:
                _safe_unlink(self._agent_path(worst.agent_id))

        new_id = self._next_id()
        entry = HallOfFameEntry(
            agent_id=new_id,
            validation_score=validation_score,
            generation=generation,
            roi=roi,
            expectancy=expectancy,
            train_fitness=train_fitness,
            quality_count=quality_count,
            total_trades=total_trades,
            val_fitness=val_fitness,
            base_combined_fitness=base_combined_fitness,
            benchmark_excess_usd=benchmark_excess_usd,
        )
        self.entries.append(entry)

        if self.hof_dir:
            _save_params(params, self._agent_path(new_id))

        return True

    # ------------------------------------------------------------------
    # Batch update (Eigen2 ``update_from_generation``)
    # ------------------------------------------------------------------

    def update_from_generation(
        self,
        candidates: List[
            Tuple[Any, float, int, float, float, float, int, int, float, float, float]
        ],
        generation: int,
    ) -> List[Tuple[int, float, str]]:
        """Two-phase admission: fill then cascading swap.

        Each candidate tuple:
            (params, combined_score, agent_idx, roi, expectancy,
             train_fitness, quality_count, total_trades, val_fitness,
             base_combined_fitness, benchmark_excess_usd)

        Returns list of ``(agent_idx, score, action_str)``.
        """
        results: List[Tuple[int, float, str]] = []
        if not candidates:
            return results

        sorted_cands = sorted(candidates, key=lambda c: c[1], reverse=True)

        # Phase 1: fill empty slots with positive-score candidates
        if not self.is_full():
            for params, score, idx, roi, exp, tf, qc, tt, vf, bcf, bh_excess in sorted_cands:
                if self.is_full():
                    break
                if score > 0:
                    new_id = self._next_id()
                    entry = HallOfFameEntry(
                        agent_id=new_id,
                        validation_score=score,
                        generation=generation,
                        roi=roi,
                        expectancy=exp,
                        train_fitness=tf,
                        quality_count=qc,
                        total_trades=tt,
                        val_fitness=vf,
                        base_combined_fitness=bcf,
                        benchmark_excess_usd=bh_excess,
                    )
                    self.entries.append(entry)
                    if self.hof_dir:
                        _save_params(params, self._agent_path(new_id))
                    results.append((idx, score, "admitted"))
                else:
                    results.append((idx, score, "rejected_negative"))

        # Phase 2: cascading swap
        admitted_ids = {r[0] for r in results if r[2] == "admitted"}
        remaining = [c for c in sorted_cands if c[2] not in admitted_ids]

        if remaining and self.is_full():
            for params, score, idx, roi, exp, tf, qc, tt, vf, bcf, bh_excess in remaining:
                current_worst = min(self.entries, key=lambda e: e.validation_score)
                if score > current_worst.validation_score:
                    old_id = current_worst.agent_id
                    self.entries.remove(current_worst)
                    if self.hof_dir:
                        _safe_unlink(self._agent_path(old_id))

                    new_id = self._next_id()
                    entry = HallOfFameEntry(
                        agent_id=new_id,
                        validation_score=score,
                        generation=generation,
                        roi=roi,
                        expectancy=exp,
                        train_fitness=tf,
                        quality_count=qc,
                        total_trades=tt,
                        val_fitness=vf,
                        base_combined_fitness=bcf,
                        benchmark_excess_usd=bh_excess,
                    )
                    self.entries.append(entry)
                    if self.hof_dir:
                        _save_params(params, self._agent_path(new_id))
                    results.append(
                        (idx, score, f"replaced_{current_worst.validation_score:.2f}")
                    )
                else:
                    results.append((idx, score, "rejected_not_better"))

        return results

    # ------------------------------------------------------------------
    # EMA score erosion (Eigen2 ``recompute_all_scores``)
    # ------------------------------------------------------------------

    def recompute_all_scores(
        self,
        current_median_roi: float,
        roi_adjustment_multiplier: float,
        min_trades_threshold: int,
        erosion_alpha: float = 0.33,
        consistency_mode: bool = False,
    ) -> int:
        """Erode historical scores toward the current ROI median using EMA.

        Returns the number of entries whose score changed meaningfully (>0.01).
        """
        if not self.entries:
            return 0

        updated = 0
        for entry in self.entries:
            old = entry.validation_score

            if consistency_mode:
                target = entry.val_fitness
            else:
                base = entry.val_fitness + min(0.0, entry.train_fitness)
                conf = (
                    min(1.0, entry.quality_count / min_trades_threshold)
                    if min_trades_threshold > 0
                    else 0.0
                )
                adj = roi_adjustment_multiplier * (entry.roi - current_median_roi) / 100.0
                target = base + adj * conf

            entry.validation_score = (1.0 - erosion_alpha) * old + erosion_alpha * target
            if abs(entry.validation_score - old) > 0.01:
                updated += 1

        return updated

    # ------------------------------------------------------------------
    # Load / sample agents
    # ------------------------------------------------------------------

    def load_params(self, agent_id: int, target: Any) -> Any:
        """Load one agent's params from disk (deserialise into *target* structure)."""
        path = self._agent_path(agent_id)
        if not path.exists():
            raise FileNotFoundError(f"HoF agent file not found: {path}")
        return _load_params(path, target)

    def get_best_entry(self) -> Optional[HallOfFameEntry]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.validation_score)

    def sample_entries(self, k: int = 1) -> List[HallOfFameEntry]:
        """Return up to *k* randomly sampled entries (without replacement)."""
        if not self.entries:
            return []
        k = min(k, len(self.entries))
        indices = np.random.choice(len(self.entries), size=k, replace=False)
        return [self.entries[i] for i in indices]

    # ------------------------------------------------------------------
    # Persistence (local JSON + optional GCS)
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist metadata to local JSON and (optionally) upload to GCS."""
        if self.hof_dir is None:
            return

        metadata = {
            "capacity": self.capacity,
            "entries": [e.to_dict() for e in self.entries],
        }
        meta_path = self.hof_dir / "hall_of_fame.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if self.cloud_sync and self.cloud_sync.provider != "local":
            cloud_json = f"{self.cloud_prefix}/hall_of_fame.json"
            self.cloud_sync.upload_file_verified(str(meta_path), cloud_json)

            for entry in self.entries:
                local_agent = self._agent_path(entry.agent_id)
                if local_agent.exists():
                    cloud_agent = (
                        f"{self.cloud_prefix}/"
                        f"hof_agent_{entry.agent_id}{AGENT_FILE_SUFFIX}"
                    )
                    self.cloud_sync.upload_file(
                        str(local_agent), cloud_agent, background=True
                    )

    def load(self) -> None:
        """Load metadata from local JSON (or download from GCS first)."""
        if self.hof_dir is None:
            return

        meta_path = self.hof_dir / "hall_of_fame.json"

        if (
            not meta_path.exists()
            and self.cloud_sync
            and self.cloud_sync.provider != "local"
        ):
            cloud_json = f"{self.cloud_prefix}/hall_of_fame.json"
            self.cloud_sync.download_file(cloud_json, str(meta_path))

        if not meta_path.exists():
            return

        with open(meta_path) as f:
            metadata = json.load(f)

        self.capacity = metadata.get("capacity", self.capacity)
        self.entries = [HallOfFameEntry.from_dict(e) for e in metadata.get("entries", [])]

        if self.cloud_sync and self.cloud_sync.provider != "local":
            for entry in self.entries:
                local_agent = self._agent_path(entry.agent_id)
                if not local_agent.exists():
                    cloud_agent = (
                        f"{self.cloud_prefix}/"
                        f"hof_agent_{entry.agent_id}{AGENT_FILE_SUFFIX}"
                    )
                    self.cloud_sync.download_file(
                        cloud_agent, str(local_agent), silent=True
                    )

        logger.info(
            "HoF loaded: %d entries (best=%.2f, worst=%.2f)",
            len(self.entries),
            max((e.validation_score for e in self.entries), default=0.0),
            min((e.validation_score for e in self.entries), default=0.0),
        )
