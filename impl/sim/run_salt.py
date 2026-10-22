"""Run the salting comparison sweep, or rebuild its summary from the raw per-request files.

    python -m sim.run_salt run                 # full sweep -> results/<date>_salt-compare/
    python -m sim.run_salt report RUN_DIR      # summary.csv from RUN_DIR/per_request/*

The raw log is one gzipped CSV per configuration with, per request, the
principal, prompt length, and the prefix hit in blocks under each system.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import re
from pathlib import Path

from harness import runmeta
from harness.synth import BLOCK_TOKENS, load_trace
from sim.run_slack import MAX_POP, MODEL_CONFIG, MODEL_REVISION, SEED, TRACES, write_summary
from sim.salt import SYSTEMS, simulate_physical, summarize, system_params
from sim.slack import ModelCost, assign, block_popularity, sessions

GRID = {
    "trace": ["conversation_trace", "toolagent_trace"],
    "assignment": ["single", "session-each", "session-uniform:10", "session-uniform:100",
                   "session-uniform:1000", "request-random:100"],
    "capacity_gb": [None, 2048, 512, 128],     # shared physical pool; None = unbounded
}


def config_id(c: dict) -> str:
    cap = "inf" if c["capacity_gb"] is None else f'{c["capacity_gb"]}GB'
    return re.sub(r"[^A-Za-z0-9_.-]", "-", f'{c["trace"]}__{c["assignment"]}__cap-{cap}')


def run(grid: dict = GRID, out: Path | None = None) -> Path:
    cost = ModelCost.from_hf_config(MODEL_CONFIG)
    block_bytes = cost.kv_bytes_per_token * BLOCK_TOKENS
    run_dir = out or runmeta.new_run_dir("salt-compare")
    (run_dir / "per_request").mkdir(parents=True, exist_ok=True)
    traces, info = {}, {}
    for t in grid["trace"]:
        rows, sha = load_trace(TRACES / f"{t}.jsonl")
        pop = block_popularity(rows)
        popular = frozenset(h for h, v in pop.items() if v > MAX_POP)
        traces[t] = (rows, sessions(rows, pop, MAX_POP), popular)
        info[t] = {"sha256": sha, "requests": len(rows), "popular_blocks": len(popular)}
    runmeta.write_meta(run_dir, {
        "tool": "sim.run_salt", "grid": grid, "systems": SYSTEMS, "max_pop": MAX_POP,
        "seed": SEED, "block_tokens": BLOCK_TOKENS, "model": MODEL_REVISION,
        "model_config_sha256": hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest(),
        "kv_bytes_per_block": block_bytes, "traces": info})

    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, values))
        rows, sess, popular = traces[c["trace"]]
        principal = assign(c["assignment"], rows, sess, SEED)
        cap = None if c["capacity_gb"] is None else c["capacity_gb"] * 2**30 // block_bytes
        hits = {s: simulate_physical(rows, principal, cap, *system_params(s, popular))
                for s in SYSTEMS}
        with gzip.open(run_dir / "per_request" / f"{config_id(c)}.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["principal", "n", *SYSTEMS])
            w.writerows(zip(principal, (r.input_length for r in rows), *(hits[s] for s in SYSTEMS)))
    report(run_dir)
    return run_dir


def report(run_dir: Path) -> Path:
    """Rebuild summary.csv from the per-request files and the model config only."""
    meta = json.loads((run_dir / "meta.json").read_text())
    if hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest() != meta["config"]["model_config_sha256"]:
        raise SystemExit("model config differs from the one this run used")
    cost = ModelCost.from_hf_config(MODEL_CONFIG)
    grid = meta["config"]["grid"]
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, values))
        with gzip.open(run_dir / "per_request" / f"{config_id(c)}.csv.gz", "rt", newline="") as f:
            rows = list(csv.DictReader(f))
        n = [int(r["n"]) for r in rows]
        hits = {s: [int(r[s]) for r in rows] for s in SYSTEMS}
        out.append({**c, "config_id": config_id(c),
                    "principals": len({r["principal"] for r in rows}),
                    **summarize(n, hits, cost)})
    write_summary(run_dir / "summary.csv", out)
    return run_dir / "summary.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    r = sub.add_parser("report")
    r.add_argument("run_dir", type=Path)
    a = ap.parse_args()
    print(run() if a.cmd == "run" else report(a.run_dir))


if __name__ == "__main__":
    main()
