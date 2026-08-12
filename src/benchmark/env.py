"""Ortam yakalama — her ölçüm dosyasına gömülür (tekrarlanabilirlik, konvansiyon gereği)."""
from __future__ import annotations

import platform
import subprocess


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def thermal_state() -> str:
    """pmset ile termal durum (sudo gerektirmez). 'nominal' = uyarı yok."""
    out = _cmd(["pmset", "-g", "therm"])
    if "No thermal warning" in out and "No performance warning" in out:
        return "nominal"
    return out[:200] or "unknown"


def capture() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "mem_bytes": int(_cmd(["sysctl", "-n", "hw.memsize"]) or 0),
        "thermal": thermal_state(),
        "git_commit": _cmd(["git", "rev-parse", "--short", "HEAD"]),
    }
    for mod in ("torch", "torchvision", "onnx", "onnxruntime", "coremltools", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info
