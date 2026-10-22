import csv

import pytest

from harness.synth import BLOCK_TOKENS as B, TraceRow
from sim import run_salt
from sim.salt import SYSTEMS, simulate_physical, summarize, system_params
from sim.slack import ModelCost

COST = ModelCost.from_hf_config(run_salt.MODEL_CONFIG)


def rows(*chains):
    return [TraceRow(i, i, len(c) * B, 1, tuple(c)) for i, c in enumerate(chains)]


def all_hits(r, principal, cap, popular=frozenset()):
    return {s: simulate_physical(r, principal, cap, *system_params(s, popular)) for s in SYSTEMS}


def test_single_principal_all_systems_equal():
    r = rows([0, 1, 2], [0, 1, 3], [0, 4])
    h = all_hits(r, [0, 0, 0], None)
    assert h["global"] == h["salt"] == h["salt-public"] == h["mirage"] == [0, 2, 1]


def test_salt_loses_cross_principal_reuse():
    r = rows([0, 1, 2], [0, 1, 3])
    h = all_hits(r, [0, 1], None)
    assert h["global"] == h["mirage"] == [0, 2]
    assert h["salt"] == [0, 0]


def test_public_set_recovers_template_only():
    # Block 0 is a public template; block 1 is shared but not public.
    r = rows([0, 1, 2], [0, 1, 3])
    h = all_hits(r, [0, 1], None, popular=frozenset({0}))
    assert h["salt-public"] == [1, 1]     # pinned template hits even on first use
    assert h["mirage"] == [1, 2]


def test_salt_duplicates_blocks_under_capacity():
    # Three principals share [7, 8]. Global keeps one copy; salt needs three.
    r = rows([7, 8], [7, 8], [7, 8], [7, 8])
    p = [0, 1, 2, 0]
    h = all_hits(r, p, cap=4)
    assert h["global"] == [0, 2, 2, 2]
    assert h["salt"] == [0, 0, 0, 0]       # principal 0's copy was evicted by 1 and 2


def test_summary_signs():
    r = rows([0, 1, 2], [0, 1, 3])
    n = [x.input_length for x in r]
    s = summarize(n, all_hits(r, [0, 1], None), COST)
    assert s["mirage_vs_salt"] > 0 and s["mirage_vs_global"] == 0


def test_run_and_report_roundtrip(tmp_path):
    if not (run_salt.TRACES / "conversation_trace.jsonl").exists():
        pytest.skip("run scripts/fetch_traces.sh")
    grid = {"trace": ["conversation_trace"], "assignment": ["single", "session-each"],
            "capacity_gb": [None]}
    out = tmp_path / "run"
    out.mkdir()
    run_salt.run(grid, out)
    first = (out / "summary.csv").read_text()
    run_salt.report(out)
    assert (out / "summary.csv").read_text() == first
    s = {r["assignment"]: r for r in csv.DictReader(open(out / "summary.csv"))}
    single = s["single"]
    # One principal: salting changes nothing. Pinning the public template saves its
    # first computation, so the pinned systems are slightly lower.
    assert single["flops_global"] == single["flops_salt"]
    assert single["flops_salt-public"] == single["flops_mirage"]
    assert int(single["flops_mirage"]) < int(single["flops_global"])
    assert float(s["session-each"]["mirage_vs_salt"]) > 0
