"""Run directories and the metadata saved alongside every raw log."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

IMPL = Path(__file__).resolve().parents[1]
RESULTS = IMPL / "results"
TRACKED_PACKAGES = ["vllm", "mooncake-transfer-engine", "aiohttp", "numpy",
                    "scikit-learn", "matplotlib"]


def new_run_dir(experiment: str, root: Path = RESULTS) -> Path:
    """Create results/<YYYY-MM-DD>_<experiment>[_N]/ and return it."""
    day = dt.date.today().isoformat()
    base = root / f"{day}_{experiment}"
    d, n = base, 1
    while d.exists():
        n += 1
        d = base.with_name(f"{base.name}_{n}")
    d.mkdir(parents=True)
    return d


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=IMPL, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict:
    out = {}
    for p in TRACKED_PACKAGES:
        try:
            out[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            out[p] = None
    return out


def collect(config: dict) -> dict:
    clock = time.get_clock_info("monotonic")
    return {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": sys.version,
        "packages": _versions(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "clock": {"source": "time.monotonic_ns", "implementation": clock.implementation,
                  "resolution_s": clock.resolution, "adjustable": clock.adjustable},
        "config": config,
    }


def write_meta(run_dir: Path, config: dict) -> dict:
    meta = collect(config)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta
