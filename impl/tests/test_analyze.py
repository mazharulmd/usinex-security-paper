import json

import numpy as np
import pytest

from harness import analyze
from harness.client import Record, RecordWriter

MEANS = {"miss": 400.0, "hbm": 40.0, "local_dram": 80.0, "remote_dram": 120.0, "ssd": 250.0}


def write_sweep(path, means, n=60, sd=0.05, seed=0, loads=("idle", "moderate")):
    """Synthetic sweep records: lognormal TTFT around a per-tier mean (ms)."""
    rng = np.random.default_rng(seed)
    w = RecordWriter(path)
    i = 0
    for load in loads:
        for tier, m in means.items():
            for _ in range(n):
                ttft = m * np.exp(rng.normal(0, sd))
                w.write(Record(f"p{i}", "attacker", "u", 4096, 1,
                               {"tier": tier, "load": load, "prompt_len": 4096},
                               t_send_ns=0, t_first_token_ns=int(ttft * 1e6), http_status=200))
                i += 1
    w.write(Record("bg", "background", "u", 10, 1, {"role": "background"},
                   t_send_ns=0, t_first_token_ns=1, http_status=200))
    w.write(Record("e", "attacker", "u", 4096, 1,
                   {"tier": "ssd", "load": "moderate", "prompt_len": 4096}, http_status=429))
    w.write(Record("u", "attacker", "u", 4096, 1,
                   {"tier": "ssd", "load": "moderate", "prompt_len": 4096, "label_ok": False},
                   t_send_ns=0, t_first_token_ns=5, http_status=200))
    w.close()


def run(tmp_path, means, **kw):
    f = tmp_path / "requests.jsonl"
    write_sweep(f, means, **kw)
    out = analyze.main([str(tmp_path), "--bootstrap", "200"])
    return json.loads((out / "gate.json").read_text()), out


def test_separable_tiers_pass_gate(tmp_path):
    g, out = run(tmp_path, MEANS)
    assert g["A1"] == "pass" and g["A2"] == "pass"
    auc = g["by_prompt_len"]["4096"]
    assert auc["miss_vs_hit"]["auc"] > 0.99 and auc["which_tier"]["auc"] > 0.95
    assert g["dropped"] == {"http_429": 1, "tier_label_unconfirmed": 1}
    assert (out / "tier_ladder.png").stat().st_size > 0
    for f in ("probes.csv", "tier_stats.csv", "auc.csv"):
        assert (out / f).exists()


def test_indistinguishable_tiers_fail_gate(tmp_path):
    # Same distribution for every tier: the check must be able to fail.
    g, _ = run(tmp_path, {t: 100.0 for t in MEANS}, sd=0.3)
    assert g["A1"] == "fail" and g["A2"] == "fail"
    auc = g["by_prompt_len"]["4096"]
    assert 0.3 < auc["miss_vs_hit"]["auc"] < 0.7 and 0.3 < auc["which_tier"]["auc"] < 0.7


def test_gate_not_measured_without_gate_load(tmp_path):
    g, _ = run(tmp_path, MEANS, loads=("idle",))
    assert g["A1"] == "not_measured" and g["A2"] == "not_measured"


def test_too_few_probes_reported_not_guessed():
    x = np.array([1.0, 2.0, 3.0])
    assert analyze.evaluate(x, np.array(["a", "b", "b"])) is None
