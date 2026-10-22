import hashlib
import json
from pathlib import Path

import pytest

from harness.synth import (BLOCK_TOKENS, SynthConfig, block_tokens, load_trace,
                           prompt_tokens, trace_namespace)

TRACES = Path(__file__).resolve().parents[1] / "data" / "traces"


def digest(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode()).hexdigest()


def test_golden_block():
    # Freezes SYNTH_VERSION 1. If this changes, every recorded run's prompts change.
    b = block_tokens("trace:conversation_trace", 0)
    assert len(b) == BLOCK_TOKENS
    assert digest(b) == GOLDEN_BLOCK0


def test_matches_written_spec():
    # Re-derive the first tokens directly from the docstring's specification.
    lo, hi = 1000, 100000
    n = hi - lo
    limit = 2**32 - (2**32 % n)
    d = hashlib.sha256(b"mirage-synth-v1|ns|0|42|" + (0).to_bytes(8, "big")).digest()
    words = [int.from_bytes(d[i:i + 4], "big") for i in range(0, 32, 4)]
    expect = [lo + w % n for w in words if w < limit]
    assert list(block_tokens("ns", 42)[:len(expect)]) == expect


def test_vocab_range():
    cfg = SynthConfig(vocab_lo=5, vocab_hi=12)
    b = block_tokens("x", 3, cfg)
    assert min(b) >= 5 and max(b) < 12
    assert len(set(b)) == 7


def test_same_hash_same_block_different_hash_different_block():
    assert block_tokens("ns", 7) == block_tokens("ns", 7)
    assert block_tokens("ns", 7) != block_tokens("ns", 8)
    assert block_tokens("ns", 7) != block_tokens("other", 7)
    assert block_tokens("ns", 7) != block_tokens("ns", 7, SynthConfig(seed=1))


def test_prompt_shares_exactly_common_blocks():
    a = prompt_tokens("ns", [0, 1, 2], 1300)
    b = prompt_tokens("ns", [0, 1, 9], 1400)
    assert len(a) == 1300 and len(b) == 1400
    assert a[:1024] == b[:1024]
    assert a[1024] != b[1024] or a[1024:1040] != b[1024:1040]


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        prompt_tokens("ns", [0, 1], 1100)


@pytest.mark.parametrize("name", ["conversation_trace", "toolagent_trace", "synthetic_trace"])
def test_real_trace_loads(name):
    path = TRACES / f"{name}.jsonl"
    if not path.exists():
        pytest.skip("run scripts/fetch_traces.sh")
    rows, sha = load_trace(path)
    assert len(sha) == 64 and rows
    ns = trace_namespace(path)
    r = rows[0]
    assert len(prompt_tokens(ns, list(r.hash_ids), r.input_length)) == r.input_length


GOLDEN_BLOCK0 = "7572519498e44d69de0939f8481248e258d026ad71a815d3d03f1e55fc3c908d"
