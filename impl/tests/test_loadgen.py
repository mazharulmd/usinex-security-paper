from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

from harness.client import Client
from harness.loadgen import replay, select_rows
from harness.mockserver import MockConfig, make_app
from harness.synth import TraceRow, load_trace, trace_namespace

TRACE = Path(__file__).resolve().parents[1] / "data" / "traces" / "conversation_trace.jsonl"


def rows_at(ms):
    return [TraceRow(i, t, 600, 3, (i, 10_000 + i)) for i, t in enumerate(ms)]


def test_select_rows():
    rows = rows_at([0, 10, 20, 30])
    assert [r.index for r in select_rows(rows, 10, 20)] == [1, 2]
    assert [r.index for r in select_rows(rows, limit=2)] == [0, 1]


async def test_replay_is_open_loop_and_on_schedule():
    # Each request takes longer (TTFT ~40 ms) than the 10 ms gap between sends,
    # so a closed loop would fall behind; an open loop must not.
    srv = TestServer(make_app(MockConfig(base_ms=40.0, tbt_ms=1.0)))
    await srv.start_server()
    try:
        async with Client(str(srv.make_url("")).rstrip("/"), "m") as c:
            recs = await replay(c, rows_at([i * 10 for i in range(20)]), "ns")
    finally:
        await srv.close()
    assert len(recs) == 20 and all(r.http_status == 200 for r in recs)
    lags_ms = [(r.t_send_ns - r.scheduled_ns) / 1e6 for r in recs]
    assert max(lags_ms) < 10
    assert all(r.labels["role"] == "background" for r in recs)


async def test_replay_real_trace_prefix_sharing():
    if not TRACE.exists():
        pytest.skip("run scripts/fetch_traces.sh")
    rows, _ = load_trace(TRACE)
    srv = TestServer(make_app(MockConfig(base_ms=1.0, miss_ms_per_token=0.0, tbt_ms=0.0)))
    await srv.start_server()
    try:
        async with Client(str(srv.make_url("")).rstrip("/"), "m") as c:
            ns = trace_namespace(TRACE)
            [r0] = await replay(c, rows[0:1], ns, max_output=1)
            [r1] = await replay(c, rows[1:2], ns, max_output=1)
    finally:
        await srv.close()
    # Rows 0 and 1 share exactly their first hash_id (one 512-token block).
    assert rows[0].hash_ids[0] == rows[1].hash_ids[0] and rows[0].hash_ids[1] != rows[1].hash_ids[1]
    assert r0.usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert r1.usage["prompt_tokens_details"]["cached_tokens"] == 512
