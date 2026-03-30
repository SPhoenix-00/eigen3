"""Wandb-style run names: ``adjective-noun-N`` with incrementing *N* per pair."""

from __future__ import annotations

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


def next_index_for_adjective_noun(checkpoint_root: Path, adj: str, noun: str) -> int:
    """Return the next free integer suffix for ``adj-noun-*`` under *checkpoint_root*."""
    pat = re.compile(rf"^{re.escape(adj)}-{re.escape(noun)}-(\d+)$")
    max_n = 0
    if checkpoint_root.is_dir():
        for p in checkpoint_root.iterdir():
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
    """Pick ``adjective-noun-N``; *N* increments for existing dirs with the same pair."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    r = rng or random.Random()
    for _ in range(256):
        adj = r.choice(_ADJECTIVES)
        noun = r.choice(_NOUNS)
        n = next_index_for_adjective_noun(checkpoint_root, adj, noun)
        name = f"{adj}-{noun}-{n}"
        if not (checkpoint_root / name).exists():
            return name
    return f"{adj}-{noun}-{n}-{r.randint(10000, 99999)}"
