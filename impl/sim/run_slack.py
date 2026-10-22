"""Run the slack simulation sweep, or rebuild its summary from the raw per-request files.

    python -m sim.run_slack run                 # full sweep -> results/<date>_slack-sim/
    python -m sim.run_slack report RUN_DIR      # summary.csv from RUN_DIR/per_request/*

The raw log of a run is one gzipped CSV per configuration with, per request,
the principal, prompt length, and p_glob / p_virt in blocks. ``report``
recomputes every summary number from those files and the model config alone.
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
from sim.slack import (ModelCost, Outcome, assign, block_popularity, sessions, simulate,
                       summarize)

IMPL = Path(__file__).resolve().parents[1]
TRACES = IMPL / "data" / "traces"
MODEL_CONFIG = IMPL / "configs" / "models" / "qwen2.5-7b-instruct.config.json"
MODEL_REVISION = "Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28"

GRID = {
    "trace": ["conversation_trace", "toolagent_trace"],
    "assignment": ["single", "session-each", "session-uniform:10", "session-uniform:100",
                   "session-uniform:1000", "request-random:100"],
    "public": ["none", "popular"],
    "capacity_gb": [None, 2048, 512],          # global pool; None = unbounded
    "quota_tokens": [None, 1_000_000],         # per-principal virtual quota; None = unbounded
}
MAX_POP = 50       # session-link / public-set popularity threshold (see sim.slack.sessions)
SEED = 0
CONTEXT_LIMIT = 32768  # Qwen2.5-7B native context, for reporting only


def config_id(c: dict) -> str:
    cap = "inf" if c["capacity_gb"] is None else f'{c["capacity_gb"]}GB'
    q = "inf" if c["quota_tokens"] is None else str(c["quota_tokens"])
    raw = f'{c["trace"]}__{c["assignment"]}__pub-{c["public"]}__cap-{cap}__quota-{q}'
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


def write_outcome(path: Path, o: Outcome) -> None:
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["principal", "n", "p_glob", "p_virt"])
        w.writerows(zip(o.principal, o.n, o.p_glob, o.p_virt))


def read_outcome(path: Path) -> Outcome:
    o = Outcome([], [], [], [])
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            o.principal.append(int(row["principal"]))
            o.n.append(int(row["n"]))
            o.p_glob.append(int(row["p_glob"]))
            o.p_virt.append(int(row["p_virt"]))
    return o


def write_summary(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run(grid: dict = GRID, out: Path | None = None) -> Path:
    cost = ModelCost.from_hf_config(MODEL_CONFIG)
    block_bytes = cost.kv_bytes_per_token * BLOCK_TOKENS
    run_dir = out or runmeta.new_run_dir("slack-sim")
    (run_dir / "per_request").mkdir(parents=True, exist_ok=True)
    traces, trace_info = {}, {}
    for t in grid["trace"]:
        rows, sha = load_trace(TRACES / f"{t}.jsonl")
        pop = block_popularity(rows)
        sess = sessions(rows, pop, MAX_POP)
        traces[t] = (rows, pop, sess)
        trace_info[t] = {"sha256": sha, "requests": len(rows), "sessions": len(set(sess)),
                         "popular_blocks": sum(1 for v in pop.values() if v > MAX_POP),
                         "distinct_blocks": len(pop),
                         "requests_over_context_limit": sum(r.input_length > CONTEXT_LIMIT
                                                            for r in rows)}
    runmeta.write_meta(run_dir, {
        "tool": "sim.run_slack", "grid": grid, "max_pop": MAX_POP, "seed": SEED,
        "block_tokens": BLOCK_TOKENS, "model": MODEL_REVISION,
        "model_config_sha256": hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest(),
        "cost_model": cost.__dict__, "kv_bytes_per_block": block_bytes,
        "context_limit": CONTEXT_LIMIT, "traces": trace_info})

    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, values))
        rows, pop, sess = traces[c["trace"]]
        principal = assign(c["assignment"], rows, sess, SEED)
        public = (frozenset(h for h, v in pop.items() if v > MAX_POP)
                  if c["public"] == "popular" else frozenset())
        cap = None if c["capacity_gb"] is None else c["capacity_gb"] * 2**30 // block_bytes
        quota = None if c["quota_tokens"] is None else c["quota_tokens"] // BLOCK_TOKENS
        o = simulate(rows, principal, cap, quota, public)
        write_outcome(run_dir / "per_request" / f"{config_id(c)}.csv.gz", o)
    report(run_dir)
    return run_dir


def report(run_dir: Path) -> Path:
    """Rebuild summary.csv from the per-request files and the model config only."""
    meta = json.loads((run_dir / "meta.json").read_text())
    grid = meta["config"]["grid"]
    cost = ModelCost.from_hf_config(MODEL_CONFIG)
    if hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest() != meta["config"]["model_config_sha256"]:
        raise SystemExit("model config differs from the one this run used")
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        c = dict(zip(keys, values))
        o = read_outcome(run_dir / "per_request" / f"{config_id(c)}.csv.gz")
        rows.append({**c, "config_id": config_id(c), **summarize(o, cost)})
    write_summary(run_dir / "summary.csv", rows)
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
