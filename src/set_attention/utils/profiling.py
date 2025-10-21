from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Any

import torch

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


@contextmanager
def profiler(enable: bool = False):
    """Context manager to profile wall time, peak VRAM, and optional GPU power.

    Yields a dict that is populated on exit: {"time_s", "gpu_peak_mem_mib", "gpu_power_w"}.
    """
    stats: Dict[str, Any] = {
        "time_s": 0.0,
        "gpu_peak_mem_mib": 0.0,
        "gpu_power_w": None,
        "gpu_util_pct": None,
        "gpu_mem_util_pct": None,
        "cpu_user_s": None,
        "cpu_sys_s": None,
        "cpu_pct": None,
        "gpu_active_s": None,
    }
    handle = None
    start_evt = None
    end_evt = None
    try:
        if enable and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # CUDA event timing for GPU active time (coarse)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
        if enable and _HAS_NVML and torch.cuda.is_available():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        if enable and _HAS_PSUTIL:
            p = psutil.Process()
            c0 = p.cpu_times()
            t0_wall = time.time()
        t0 = time.time()
        if start_evt is not None:
            torch.cuda.synchronize()
            start_evt.record()
        yield stats
    finally:
        if end_evt is not None:
            end_evt.record()
            torch.cuda.synchronize()
            try:
                stats["gpu_active_s"] = float(start_evt.elapsed_time(end_evt)) / 1000.0
            except Exception:
                stats["gpu_active_s"] = None
        stats["time_s"] = time.time() - t0
        if enable and torch.cuda.is_available():
            stats["gpu_peak_mem_mib"] = float(torch.cuda.max_memory_allocated()) / (1024**2)
        if enable and _HAS_NVML and handle is not None:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                stats["gpu_power_w"] = float(power_mw) / 1000.0
            except Exception:
                stats["gpu_power_w"] = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                # util.gpu and util.memory are percentages (0-100)
                stats["gpu_util_pct"] = float(util.gpu)
                stats["gpu_mem_util_pct"] = float(util.memory)
            except Exception:
                pass
        if enable and _HAS_PSUTIL:
            try:
                p = psutil.Process()
                c1 = p.cpu_times()
                user = getattr(c1, "user", 0.0) - getattr(c0, "user", 0.0)
                system = getattr(c1, "system", 0.0) - getattr(c0, "system", 0.0)
                stats["cpu_user_s"] = user
                stats["cpu_sys_s"] = system
                # Approximate CPU percentage over the profiled wall time
                wall = max(1e-6, time.time() - t0)
                stats["cpu_pct"] = 100.0 * (user + system) / wall
            except Exception:
                pass
        if handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
