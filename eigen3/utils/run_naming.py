"""Wandb-style run names: ``adjective-noun-N`` with a **global** incrementing *N*.

*N* is one greater than the largest numeric suffix among existing run directories
under the checkpoint root (not per adjective–noun pair), so successive launches
get ``…-1``, ``…-2``, … even when the random words change each time.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

# Short, filesystem-friendly tokens (lowercase, no spaces).
_ADJECTIVES: tuple[str, ...] = (
    "swift",
    "calm",
    "bright",
    "cosmic",
    "gentle",
    "noble",
    "quiet",
    "rapid",
    "silent",
    "brave",
    "clever",
    "daring",
    "eager",
    "lucky",
    "mighty",
    "proud",
    "sunny",
    "vivid",
    "wild",
    "zen",
    "azure",
    "crimson",
    "golden",
    "silver",
    "atomic",
    "crystal",
    "electric",
    "frozen",
    "jade",
    "amber",
    "ivory",
    "lunar",
    "solar",
    "stellar",
    "ocean",
    "forest",
    "desert",
    "winter",
    "summer",
    "spring",
    "autumn",
    "velvet",
    "rustic",
    "mystic",
    "radiant",
)

def resolved_checkpoint_root(artifact_root: Path) -> Path:
    """Directory used for run folders and suffix scanning.

    Uses a **stable** path across Hydra output dirs: when neither
    ``EIGEN3_CHECKPOINT_ROOT`` nor ``EIGEN3_ARTIFACT_ROOT`` is set, this is
    ``<hydra original cwd>/checkpoints`` (repo-level), not the per-job Hydra
    working directory (which would be empty on every new launch).
    """
    raw = os.environ.get("EIGEN3_CHECKPOINT_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    if os.environ.get("EIGEN3_ARTIFACT_ROOT"):
        return (artifact_root / "checkpoints").resolve()
    try:
        from hydra.utils import get_original_cwd

        return (Path(get_original_cwd()) / "checkpoints").resolve()
    except Exception:
        return (artifact_root / "checkpoints").resolve()


_NOUNS: tuple[str, ...] = (
    "thunder",
    "river",
    "mountain",
    "meadow",
    "harbor",
    "canyon",
    "glacier",
    "comet",
    "nebula",
    "falcon",
    "tiger",
    "dragon",
    "phoenix",
    "raven",
    "wolf",
    "eagle",
    "castle",
    "tower",
    "bridge",
    "harvest",
    "ember",
    "frost",
    "breeze",
    "summit",
    "delta",
    "vertex",
    "matrix",
    "vector",
    "tensor",
    "signal",
    "pulse",
    "beacon",
    "anchor",
    "compass",
    "horizon",
    "prism",
    "quartz",
    "cipher",
    "ledger",
    "oracle",
    "voyage",
    "coral",
    "sapphire",
    "granite",
    "cedar",
    "willow",
    "maple",
)


_RUN_DIR_SUFFIX = re.compile(r"^[a-z]+-[a-z]+-(\d+)$")


def next_global_run_suffix(checkpoint_root: Path) -> int:
    """Return ``1 + max(N)`` for every existing ``<word>-<word>-N`` run directory."""
    root = checkpoint_root.resolve()
    max_n = 0
    if root.is_dir():
        for p in root.iterdir():
            if not p.is_dir() or p.name.startswith("."):
                continue
            m = _RUN_DIR_SUFFIX.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def next_index_for_adjective_noun(checkpoint_root: Path, adj: str, noun: str) -> int:
    """Return the next free integer suffix for ``adj-noun-*`` under *checkpoint_root*."""
    root = checkpoint_root.resolve()
    pat = re.compile(rf"^{re.escape(adj)}-{re.escape(noun)}-(\d+)$")
    max_n = 0
    if root.is_dir():
        for p in root.iterdir():
            if p.is_dir():
                m = pat.match(p.name)
                if m:
                    max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def generate_wandb_style_run_name(
    checkpoint_root: Path,
    *,
    rng: random.Random | None = None,
) -> str:
    """Pick random ``adjective`` and ``noun``; suffix *N* is global (max existing + 1)."""
    root = checkpoint_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    r = rng or random.Random()
    n = next_global_run_suffix(root)
    for _ in range(512):
        adj = r.choice(_ADJECTIVES)
        noun = r.choice(_NOUNS)
        name = f"{adj}-{noun}-{n}"
        if not (root / name).exists():
            return name
        n += 1
    return f"{adj}-{noun}-{n}-{r.randint(10000, 99999)}"
