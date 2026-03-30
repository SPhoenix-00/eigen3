"""High-visibility terminal banners (ANSI colors when supported)."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eigen3.erl.cloud_sync import CloudSync

# Tee mirror (Eigen2 compat): avoid ANSI in the log file.
_TEE_ENV = "EIGEN3_TRAINING_LOG"


def _use_color() -> bool:
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("EIGEN3_NO_COLOR", "").strip():
        return False
    if os.environ.get(_TEE_ENV, "").strip():
        return False
    if os.environ.get("TERM", "").lower() in ("dumb", ""):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class _C:
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    FG_CYAN = "\033[96m"
    FG_GREEN = "\033[92m"
    FG_YELLOW = "\033[93m"
    FG_MAGENTA = "\033[95m"
    FG_WHITE = "\033[97m"


def _paint(text: str, *codes: str) -> str:
    if not _use_color():
        return text
    return f"{''.join(codes)}{text}{_C.RST}"


def print_training_identity_banner(
    *,
    run_name: str,
    cloud_sync: CloudSync,
    cloud_prefix: str,
    checkpoint_dir: str,
    resume: bool,
) -> None:
    """Print run name and cloud mode before Phase 1 (stdout; respects NO_COLOR / tee)."""
    bar = "═" * 62
    mode = "RESUME" if resume else "NEW RUN"

    if cloud_sync.provider == "local":
        cloud_line = _paint("DISABLED", _C.BOLD, _C.FG_YELLOW)
        cloud_detail = "local-only (no GCS uploads)"
    else:
        cloud_line = _paint("ENABLED", _C.BOLD, _C.FG_GREEN)
        b = cloud_sync.bucket_name or "?"
        cloud_detail = f"gs://{b}  |  HoF prefix: {cloud_prefix}"

    run_painted = _paint(run_name, _C.BOLD, _C.FG_CYAN)
    mode_painted = _paint(mode, _C.BOLD, _C.FG_MAGENTA)

    lines = [
        "",
        _paint(bar, _C.DIM),
        f"  {_paint('RUN NAME', _C.BOLD, _C.FG_WHITE)}    {run_painted}",
        f"  {_paint('MODE', _C.BOLD, _C.FG_WHITE)}       {mode_painted}",
        f"  {_paint('CLOUD', _C.BOLD, _C.FG_WHITE)}       {cloud_line}  —  {cloud_detail}",
        f"  {_paint('CHECKPOINTS', _C.BOLD, _C.FG_WHITE)} {checkpoint_dir}",
        _paint(bar, _C.DIM),
        "",
    ]
    print("\n".join(lines), flush=True)
