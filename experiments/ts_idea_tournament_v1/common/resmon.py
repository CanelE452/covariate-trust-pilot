"""Standalone resource monitor. Runs as its own process, independent of the
training parent, and writes a breach marker the training loop polls.

Usage: python resmon.py <parent_pid> <out_dir>
Limits: process-tree RSS <= 10 GB, system available RAM >= 8 GB,
        system memory usage <= 80%, GPU used <= 9.5 GB.
"""
from __future__ import annotations

import json
import os
import sys
import time

RSS_CAP_GB = 10.0
AVAIL_FLOOR_GB = 8.0
SYS_PCT_CAP = 80.0
GPU_CAP_GB = 9.5
INTERVAL_S = 5.0


def gpu_used_gb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip().splitlines()
        return float(out[0]) / 1024.0
    except Exception:
        return None


def tree_rss_gb(proc):
    import psutil
    tot = 0
    try:
        tot += proc.memory_info().rss
        for c in proc.children(recursive=True):
            try:
                tot += c.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass
    return tot / 1e9


def main():
    import psutil
    parent_pid = int(sys.argv[1])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "resource_trace.jsonl")
    breach = os.path.join(out, "MEMORY_CAP_BREACH.json")
    peak = {"tree_rss_gb": 0.0, "gpu_used_gb": 0.0, "sys_pct": 0.0}
    try:
        proc = psutil.Process(parent_pid)
    except Exception:
        return
    with open(log, "a", buffering=1) as f:
        while True:
            if not proc.is_running():
                break
            vm = psutil.virtual_memory()
            rec = {
                "t": time.time(),
                "tree_rss_gb": round(tree_rss_gb(proc), 3),
                "sys_avail_gb": round(vm.available / 1e9, 2),
                "sys_pct": vm.percent,
                "gpu_used_gb": gpu_used_gb(),
            }
            f.write(json.dumps(rec) + "\n")
            peak["tree_rss_gb"] = max(peak["tree_rss_gb"], rec["tree_rss_gb"])
            peak["sys_pct"] = max(peak["sys_pct"], rec["sys_pct"])
            if rec["gpu_used_gb"]:
                peak["gpu_used_gb"] = max(peak["gpu_used_gb"], rec["gpu_used_gb"])
            viol = []
            if rec["tree_rss_gb"] > RSS_CAP_GB:
                viol.append("tree_rss")
            if rec["sys_avail_gb"] < AVAIL_FLOOR_GB:
                viol.append("sys_available")
            if rec["sys_pct"] > SYS_PCT_CAP:
                viol.append("sys_percent")
            if rec["gpu_used_gb"] and rec["gpu_used_gb"] > GPU_CAP_GB:
                viol.append("gpu_used")
            if viol:
                with open(breach, "w") as bf:
                    json.dump({"violations": viol, **rec}, bf, indent=1)
                break
            with open(os.path.join(out, "resource_peak.json"), "w") as pf:
                json.dump(peak, pf, indent=1)
            time.sleep(INTERVAL_S)
    with open(os.path.join(out, "resource_peak.json"), "w") as pf:
        json.dump(peak, pf, indent=1)


if __name__ == "__main__":
    main()
