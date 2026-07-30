"""Environment audit.

Runs no model inference and generates no data.  Everything reported here is
observed from the running interpreter, not assumed from documentation.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path

MODEL_CACHE_DIRNAME = "models--amazon--chronos-2"


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def _run_ok(cmd: list[str]) -> tuple[bool, str]:
    """Return (succeeded, output).  Needed because `git rev-parse HEAD` echoes its
    argument on an unborn branch instead of failing loudly on stdout."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.returncode == 0, (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return False, f"<unavailable: {type(exc).__name__}: {exc}>"


def _version(pkg: str) -> str:
    try:
        return md.version(pkg)
    except Exception:  # noqa: BLE001
        return "<not installed>"


def git_info(root: Path) -> dict:
    inside = _run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"]) == "true"
    if not inside:
        return {"is_repo": False, "commit": "NOT_A_REPO", "status": ""}
    ok, commit = _run_ok(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"])
    if not ok or not commit or not all(c in "0123456789abcdef" for c in commit):
        commit = "UNBORN"
    return {
        "is_repo": True,
        "commit": commit,
        "branch": _run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _run(["git", "-C", str(root), "status", "--porcelain"]),
    }


def internet_available(timeout: float = 8.0) -> bool:
    import urllib.request
    for url in ("https://huggingface.co", "https://pypi.org/simple/"):
        try:
            urllib.request.urlopen(url, timeout=timeout)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def model_cache_state(hf_home: str, model_id: str = "amazon/chronos-2") -> dict:
    cache = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
    files = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file()) if cache.exists() else []
    return {
        "path": str(cache),
        "exists": cache.exists(),
        "n_files": len(files),
        "bytes": sum((cache / f).stat().st_size for f in files) if files else 0,
    }


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or _run(["bash", "-lc",
                                                   "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1024**3, 2)
    except Exception:  # noqa: BLE001
        info["ram_total_gb"] = None
    info["nvidia_smi"] = _run(["nvidia-smi",
                               "--query-gpu=name,driver_version,memory.total,memory.used",
                               "--format=csv,noheader"])
    return info


def torch_info() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    info = {
        "available": True,
        "version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "path": torch.__file__,
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_capability"] = f"{props.major}.{props.minor}"
    return info


def chronos_api_info() -> dict:
    try:
        from .chronos_adapter import api_report
        return {"importable": True, **api_report()}
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}


def collect(root: Path, hf_home: str) -> dict:
    return {
        "project_root": str(Path(root).resolve()),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_full": sys.version,
            "prefix": sys.prefix,
            "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "hardware": hardware(),
        "torch": torch_info(),
        "packages": {name: _version(name) for name in
                     ("chronos-forecasting", "transformers", "huggingface-hub", "accelerate",
                      "pandas", "numpy", "scipy", "matplotlib", "pyyaml", "typer", "rich",
                      "psutil", "pyarrow", "pytest")},
        "git": git_info(Path(root)),
        "hf_home": hf_home,
        "internet_available": internet_available(),
        "model_cache": model_cache_state(hf_home),
        "chronos_api": chronos_api_info(),
    }


def blocking_problems(audit: dict) -> list[str]:
    """Conditions that must stop the pipeline before any Chronos work."""
    problems = []
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        problems.append("BLOCKED_PYTHON_VERSION")
    t = audit["torch"]
    if not t.get("available"):
        problems.append("BLOCKED_CHRONOS_ENV: torch is not importable")
    api = audit["chronos_api"]
    if not api.get("importable"):
        problems.append(f"BLOCKED_CHRONOS_ENV: {api.get('error')}")
    elif not api.get("cross_learning_supported"):
        problems.append("BLOCKED_CHRONOS_ENV: predict_df has no cross_learning argument")
    return problems


def environment_text(audit: dict) -> str:
    lines = ["# environment", ""]

    def emit(prefix: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                emit(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            lines.append(f"{prefix}: {obj}")

    emit("", audit)
    return "\n".join(lines) + "\n"
