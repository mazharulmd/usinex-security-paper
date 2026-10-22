# Mirage prototype, phase 1

Client-side measurement harness for the week-1 tier-ladder experiment. The
serving stack is not set up yet (see `VERSIONS.md`); findings go in `REPORT.md`.

```
make venv traces test                         # set up, fetch pinned traces, run tests
python -m harness.loadgen --url ... --model ... --trace data/traces/conversation_trace.jsonl --speedup 2
make analyze RUN=results/<date>_<experiment>  # CSVs, gate.json, tier_ladder.png from raw records
```

| Module | Does |
| --- | --- |
| `harness/synth.py` | Trace `hash_ids` → token-ID prompts; the exact specification is in its docstring |
| `harness/client.py` | Streaming client; per-request timing, status, headers, full usage object |
| `harness/loadgen.py` | Open-loop trace replay at a speedup factor |
| `harness/analyze.py` | Per-tier TTFT stats, cross-validated miss-vs-hit and which-tier AUC, week-1 gate |
| `harness/runmeta.py` | `results/<date>_<experiment>/meta.json`: command, git commit, versions, clock source |
| `sim/slack.py`, `sim/run_slack.py` | Week-3 trace-driven slack simulation (`make slack`, `make slack-report RUN=...`) |
| `harness/mockserver.py` | GPU-free stand-in for tests only; nothing measured against it is a result |

Not written yet (need the testbed to design against real behaviour): `tiers.py`
(prime/evict/probe recipes and ground-truth tier labels from the Mooncake tier
log), `sweep.py`, and the launch scripts in `configs/`.
