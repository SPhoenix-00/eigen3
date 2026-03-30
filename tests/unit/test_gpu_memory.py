"""GPU memory logging helpers."""

from eigen3.utils.gpu_memory import (
    format_gpu_memory_report,
    should_log_gpu_memory_this_generation,
)


def test_should_log_gpu_memory_interval_zero():
    assert should_log_gpu_memory_this_generation(0, 0) is True
    assert should_log_gpu_memory_this_generation(0, 1) is False


def test_should_log_gpu_memory_interval_one():
    assert should_log_gpu_memory_this_generation(1, 0) is True
    assert should_log_gpu_memory_this_generation(1, 99) is True


def test_should_log_gpu_memory_interval_n():
    assert should_log_gpu_memory_this_generation(5, 0) is True
    assert should_log_gpu_memory_this_generation(5, 1) is False
    assert should_log_gpu_memory_this_generation(5, 4) is True  # 5th generation
    assert should_log_gpu_memory_this_generation(5, 9) is True  # 10th


def test_format_gpu_memory_report_runs():
    # CPU CI: JAX stats may be None; must not raise.
    s = format_gpu_memory_report()
    assert "JAX memory_stats" in s
    assert "nvidia-smi" in s
