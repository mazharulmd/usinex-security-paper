from pathlib import Path

import pytest

from harness.synth import BLOCK_TOKENS as B, TraceRow
from sim import run_slack
from sim.slack import (LRU, ModelCost, assign, block_popularity, sessions, simulate,
                       summarize)

COST = ModelCost.from_hf_config(run_slack.MODEL_CONFIG)


def rows(*chains, n_blocks_extra=0):
    """Trace rows whose input covers exactly their blocks."""
    return [TraceRow(i, i, len(c) * B + n_blocks_extra, 1, tuple(c)) for i, c in enumerate(chains)]


def test_qwen_cost_model():
    # Qwen2.5 report: 7.61B params, 6.53B non-embedding. The linear weights here
    # exclude norms and biases, so they should be just under 6.53B.
    assert 6.50e9 < COST.linear_weights < 6.53e9
    assert COST.kv_bytes_per_token == 2 * 28 * 4 * 128 * 2


def test_prefill_flops_additive_and_last_token():
    n, p = 5000, 2048
    assert COST.prefill_flops(n, 0) == COST.prefill_flops(p, 0) + COST.prefill_flops(n, p)
    assert COST.prefill_flops(n, n) == COST.prefill_flops(n, n - 1) > 0


def test_lru_evicts_oldest_and_refreshes():
    c = LRU(2)
    c.touch([1, 2])
    c.touch([1])
    c.touch([3])
    assert 1 in c and 3 in c and 2 not in c


def test_single_principal_has_no_slack():
    r = rows([0, 1, 2], [0, 1, 3], [0, 4])
    o = simulate(r, [0, 0, 0], None, None, frozenset())
    assert o.p_glob == o.p_virt == [0, 2, 1]
    assert summarize(o, COST)["slack_flops"] == 0


def test_cross_principal_hit_is_slack():
    r = rows([0, 1, 2], [0, 1, 3])
    o = simulate(r, [0, 1], None, None, frozenset())
    assert o.p_glob == [0, 2] and o.p_virt == [0, 0]
    s = summarize(o, COST)
    expect = COST.prefill_flops(3 * B, 0) - COST.prefill_flops(3 * B, 2 * B)
    assert s["slack_flops"] == s["slack_flops_positive"] == expect
    assert s["requests_with_slack"] == 1


def test_public_blocks_are_not_slack():
    r = rows([0, 1, 2], [0, 1, 3])
    o = simulate(r, [0, 1], None, None, frozenset({0, 1}))
    assert o.p_virt == [2, 2] and summarize(o, COST)["slack_flops"] == 0


def test_partial_last_block_never_reused():
    r = [TraceRow(0, 0, 2 * B + 10, 1, (0, 1, 2)), TraceRow(1, 1, 2 * B + 10, 1, (0, 1, 2))]
    o = simulate(r, [0, 1], None, None, frozenset())
    assert o.p_glob == [0, 2]


def test_virtual_hit_after_physical_eviction_is_counted():
    r = rows([0, 1], [5, 6], [0, 1])
    o = simulate(r, [0, 1, 0], capacity_blocks=2, quota_blocks=None, public=frozenset())
    assert o.p_glob[2] == 0 and o.p_virt[2] == 2
    s = summarize(o, COST)
    assert s["requests_virtual_hit_physical_miss"] == 1 and s["slack_flops"] < 0


def test_sessions_link_on_rare_blocks_only():
    # Block 0 is a template in every request; 7 and 9 continue rare histories.
    r = rows([0, 7, 1], [0, 7, 2], [0, 9], [0, 9, 3], [0, 4])
    pop = block_popularity(r)
    s = sessions(r, pop, max_pop=2)
    assert s[0] == s[1] and s[2] == s[3] and len({s[0], s[2], s[4]}) == 3


def test_assignments():
    r = rows([0], [0], [0])
    sess = [0, 0, 1]
    assert assign("single", r, sess, 0) == [0, 0, 0]
    assert assign("session-each", r, sess, 0) == sess
    u = assign("session-uniform:10", r, sess, 0)
    assert u[0] == u[1]
    assert len(assign("request-random:5", r, sess, 0)) == 3
    with pytest.raises(ValueError):
        assign("nope", r, sess, 0)


def test_run_and_report_roundtrip(tmp_path, monkeypatch):
    if not (run_slack.TRACES / "conversation_trace.jsonl").exists():
        pytest.skip("run scripts/fetch_traces.sh")
    grid = {"trace": ["conversation_trace"], "assignment": ["single", "request-random:100"],
            "public": ["none"], "capacity_gb": [None], "quota_tokens": [None]}
    out = tmp_path / "run"
    out.mkdir()
    run_slack.run(grid, out)
    first = (out / "summary.csv").read_text()
    run_slack.report(out)
    assert (out / "summary.csv").read_text() == first
    import csv
    s = {r["assignment"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    assert int(s["single"]["slack_flops"]) == 0
    assert int(s["request-random:100"]["slack_flops"]) > 0
