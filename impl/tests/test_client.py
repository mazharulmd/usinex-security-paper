import json

import pytest
from aiohttp.test_utils import TestServer

from harness import runmeta
from harness.client import Client, RecordWriter, Request, read_records
from harness.mockserver import MockConfig, make_app
from harness.synth import prompt_tokens


@pytest.fixture
async def server():
    srv = TestServer(make_app(MockConfig(tbt_ms=1.0)))
    await srv.start_server()
    yield srv
    await srv.close()


def base_url(srv):
    return str(srv.make_url("")).rstrip("/")


async def test_records_timing_tokens_and_usage(server):
    p = prompt_tokens("ns", [0, 1], 1000)
    async with Client(base_url(server), "m") as c:
        miss = await c.send(Request("r1", p, max_tokens=5, principal="alice"))
        hit = await c.send(Request("r2", p, max_tokens=5, principal="bob"))
    for r in (miss, hit):
        assert r.http_status == 200 and r.error is None
        assert len(r.token_times_ns) == 5
        assert r.t_send_ns < r.t_headers_ns <= r.t_first_token_ns == r.token_times_ns[0]
        assert r.token_times_ns == sorted(r.token_times_ns) and r.t_done_ns >= r.token_times_ns[-1]
        assert r.finish_reason == "length"
    assert miss.usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert hit.usage["prompt_tokens_details"]["cached_tokens"] == 992   # 62 full 16-token blocks
    assert hit.ttft_ns < miss.ttft_ns


async def test_cache_salt_separates(server):
    p = prompt_tokens("ns", [5], 512)
    async with Client(base_url(server), "m") as c:
        await c.send(Request("a", p, 1, cache_salt="alice"))
        r = await c.send(Request("b", p, 1, cache_salt="bob"))
    assert r.usage["prompt_tokens_details"]["cached_tokens"] == 0


async def test_non_200_is_recorded():
    srv = TestServer(make_app(MockConfig(max_inflight=0)))
    await srv.start_server()
    try:
        async with Client(base_url(srv), "m") as c:
            r = await c.send(Request("x", [1, 2, 3], 1))
    finally:
        await srv.close()
    assert r.http_status == 429 and r.t_first_token_ns is None and r.ttft_ns is None
    assert "too many" in r.error


async def test_connection_error_is_recorded():
    async with Client("http://127.0.0.1:9", "m", timeout_s=2) as c:
        r = await c.send(Request("x", [1], 1))
    assert r.http_status is None and r.error


def test_writer_roundtrip(tmp_path):
    from harness.client import Record
    w = RecordWriter(tmp_path / "r.jsonl")
    w.write(Record("a", "p", "u", 3, 1, {"tier": "miss"}, t_send_ns=10, t_first_token_ns=25))
    w.close()
    [d] = read_records(tmp_path / "r.jsonl")
    assert d["ttft_ns"] == 15 and d["labels"] == {"tier": "miss"}


def test_run_dir_and_meta(tmp_path):
    d1 = runmeta.new_run_dir("exp", root=tmp_path)
    d2 = runmeta.new_run_dir("exp", root=tmp_path)
    assert d1 != d2 and d1.name.endswith("_exp") and d2.name.endswith("_exp_2")
    meta = runmeta.write_meta(d1, {"k": 1})
    on_disk = json.loads((d1 / "meta.json").read_text())
    assert on_disk["config"] == {"k": 1} and meta["clock"]["source"] == "time.monotonic_ns"
    assert on_disk["git_commit"]
