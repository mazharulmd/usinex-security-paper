"""Open-loop background load: replay a Mooncake trace against an endpoint.

Row i is sent at  start + (timestamp_ms[i] - timestamp_ms[first]) / speedup,
whether or not earlier requests have finished (open loop), with a prompt from
synth.py and ``max_tokens = min(output_length, max_output)``. The intended send
time is stored in each record (``scheduled_ns``), so client-side lag is visible.
Load levels are set by ``speedup``; the mapping from level to speedup is fixed
per testbed by a capacity measurement, not here.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from . import runmeta
from .client import Client, Record, RecordWriter, Request
from .synth import SynthConfig, TraceRow, load_trace, prompt_tokens, trace_namespace


def select_rows(rows: list[TraceRow], start_ms: int = 0, duration_ms: int | None = None,
                limit: int | None = None) -> list[TraceRow]:
    out = [r for r in rows if r.timestamp_ms >= start_ms
           and (duration_ms is None or r.timestamp_ms < start_ms + duration_ms)]
    return out[:limit] if limit is not None else out


async def replay(client: Client, rows: list[TraceRow], namespace: str, *,
                 speedup: float = 1.0, max_output: int = 512, principal: str = "background",
                 synth: SynthConfig = SynthConfig(), on_record=None,
                 stop: asyncio.Event | None = None) -> list[Record]:
    """Send ``rows`` open-loop; return records in completion order."""
    if not rows:
        return []
    t0_trace = rows[0].timestamp_ms
    start = time.monotonic_ns()
    tasks: list[asyncio.Task] = []
    done: list[Record] = []

    async def one(row: TraceRow, at_ns: int) -> None:
        req = Request(f"bg-{row.index}", prompt_tokens(namespace, list(row.hash_ids),
                                                       row.input_length, synth),
                      max_tokens=max(1, min(row.output_length, max_output)),
                      principal=principal, scheduled_ns=at_ns,
                      labels={"role": "background", "trace": namespace, "row": row.index})
        rec = await client.send(req)
        done.append(rec)
        if on_record:
            on_record(rec)

    for row in rows:
        if stop is not None and stop.is_set():
            break
        at = start + int((row.timestamp_ms - t0_trace) * 1e6 / speedup)
        delay = (at - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(row, at)))
    await asyncio.gather(*tasks)
    return done


async def _main(a: argparse.Namespace) -> Path:
    rows, sha = load_trace(a.trace)
    rows = select_rows(rows, a.start_ms, a.duration_ms, a.limit)
    synth = SynthConfig(seed=a.seed)
    run = runmeta.new_run_dir(a.experiment)
    runmeta.write_meta(run, {"tool": "loadgen", "url": a.url, "model": a.model,
                             "trace": str(a.trace), "trace_sha256": sha,
                             "rows": len(rows), "start_ms": a.start_ms,
                             "duration_ms": a.duration_ms, "speedup": a.speedup,
                             "max_output": a.max_output, "synth": synth.as_dict()})
    w = RecordWriter(run / "requests.jsonl")
    try:
        async with Client(a.url, a.model) as c:
            await replay(c, rows, trace_namespace(a.trace), speedup=a.speedup,
                         max_output=a.max_output, synth=synth, on_record=w.write)
    finally:
        w.close()
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a Mooncake trace open-loop.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--start-ms", type=int, default=0)
    ap.add_argument("--duration-ms", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-output", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", default="loadgen")
    print(asyncio.run(_main(ap.parse_args())))


if __name__ == "__main__":
    main()
