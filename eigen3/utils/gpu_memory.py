"""GPU VRAM introspection: JAX device stats plus optional ``nvidia-smi`` totals."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


def _format_bytes(n: float) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def _format_stat_value(key: str, value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return f"{key}: {value}"
    if isinstance(value, (int, float)):
        fv = float(value)
        if fv < 0:
            return f"{key}: {value}"
        if "byte" in key.lower() or fv >= 1_048_576:
            try:
                return f"{key}: {int(fv)} ({_format_bytes(fv)})"
            except (OverflowError, ValueError):
                return f"{key}: {value}"
        if fv == int(fv):
            return f"{key}: {int(fv)}"
        return f"{key}: {value}"
    return f"{key}: {value}"


def jax_device_memory_stats(device: Optional[jax.Device] = None) -> Optional[Dict[str, Any]]:
    """Return ``device.memory_stats()`` for the default JAX GPU, or ``None`` if unavailable."""
    if device is None:
        devs = jax.local_devices()
        if not devs:
            return None
        device = devs[0]
    stats = device.memory_stats()
    if stats is None:
        return None
    return dict(stats)


def nvidia_smi_gpu_mib() -> Optional[Dict[str, float]]:
    """Parse ``nvidia-smi`` MiB used/total/free for the first GPU; ``None`` if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 3:
        return None
    used, total, free = map(float, parts[:3])
    return {"used_mib": used, "total_mib": total, "free_mib": free}


def format_gpu_memory_report() -> str:
    """Compact multi-line string for logs (JAX stats + optional ``nvidia-smi``)."""
    lines: list[str] = []
    stats = jax_device_memory_stats()
    if stats:
        lines.append("JAX memory_stats (default device):")
        for k in sorted(stats.keys()):
            lines.append(f"  {_format_stat_value(k, stats[k])}")
    else:
        lines.append("JAX memory_stats: (none — CPU backend or driver not reporting)")

    smi = nvidia_smi_gpu_mib()
    if smi:
        u, t, f = smi["used_mib"], smi["total_mib"], smi["free_mib"]
        pct = 100.0 * u / t if t > 0 else 0.0
        lines.append(
            f"nvidia-smi: {u:.0f} / {t:.0f} MiB used ({pct:.1f}%), free {f:.0f} MiB"
        )
    else:
        lines.append("nvidia-smi: (not available)")

    return "\n".join(lines)


def sync_gpu_memory_read() -> None:
    """Block until pending work completes so VRAM reads are meaningful."""
    jax.block_until_ready(jnp.array(0.0))


def log_gpu_memory_report(log: logging.Logger, tag: str) -> None:
    """Log a VRAM snapshot at INFO after syncing GPU work."""
    sync_gpu_memory_read()
    log.info("GPU memory [%s]\n%s", tag, format_gpu_memory_report())


def should_log_gpu_memory_this_generation(interval: int, gen: int) -> bool:
    """Whether to log after generation ``gen`` (0-based).

    * ``interval <= 0`` — log only after the first completed generation (``gen == 0``).
    * ``interval == 1`` — log after every generation.
    * ``interval > 1`` — log after the first generation and whenever ``(gen + 1) % interval == 0``.
    """
    if interval <= 0:
        return gen == 0
    if interval == 1:
        return True
    return gen == 0 or (gen + 1) % interval == 0
