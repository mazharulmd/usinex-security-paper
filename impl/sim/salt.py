"""Physical prefill work under global sharing, per-principal salting, and Mirage.

All systems share one physical LRU pool of the same capacity; they differ only
in how blocks are keyed:

  global       key = block                    (unrestricted sharing; insecure ceiling)
  salt         key = (principal, block)       (vLLM cache_salt on every request)
  salt-public  key = block if public, else (principal, block); public blocks pinned
               (salting with an unsalted operator-declared public set)
  mirage       key = block; public blocks pinned
               (Mirage keeps physical sharing global: goal G2)

What each principal *observes* is not modelled here (see slack.py); only the
GPU work the pool lets each system skip. Simulation rules are as in slack.py:
trace order, insertion at arrival, full 512-token blocks only, LRU with refresh
on hit, last token always computed. Pinned public blocks sit outside the LRU
capacity (the public sets used here are at most 18 blocks, about 0.5 GB of KV).
"""

from __future__ import annotations

from harness.synth import BLOCK_TOKENS, TraceRow
from sim.slack import LRU, ModelCost, prefix_hits

SYSTEMS = ["global", "salt", "salt-public", "mirage"]


def simulate_physical(rows: list[TraceRow], principal: list[int], capacity_blocks: int | None,
                      salted: bool, public: frozenset[int]) -> list[int]:
    """Prefix hit length in blocks per request for one keying scheme."""
    pool = LRU(capacity_blocks)
    hits = []
    for r, p in zip(rows, principal):
        chain = r.hash_ids[: r.input_length // BLOCK_TOKENS]
        keys = [h if (h in public or not salted) else (p, h) for h in chain]
        hits.append(prefix_hits(keys, lambda k: k in public or k in pool))
        pool.touch(k for k in keys if k not in public)
    return hits


def system_params(system: str, popular: frozenset[int]) -> tuple[bool, frozenset[int]]:
    """(salted, pinned public set) for each system."""
    return {"global": (False, frozenset()),
            "salt": (True, frozenset()),
            "salt-public": (True, popular),
            "mirage": (False, popular)}[system]


def prefill_flops(n: list[int], hits: list[int], cost: ModelCost) -> int:
    return sum(cost.prefill_flops(m, min(k * BLOCK_TOKENS, m - 1)) for m, k in zip(n, hits))


def summarize(n: list[int], hits: dict[str, list[int]], cost: ModelCost) -> dict:
    f = {s: prefill_flops(n, hits[s], cost) for s in SYSTEMS}
    f0 = sum(cost.prefill_flops(m, 0) for m in n)
    out = {"requests": len(n), "flops_nocache": f0}
    for s in SYSTEMS:
        out[f"flops_{s}"] = f[s]
        out[f"saving_{s}"] = (f0 - f[s]) / f0 if f0 else 0.0
    # Positive = Mirage does less prefill work than the baseline.
    out["mirage_vs_salt"] = (f["salt"] - f["mirage"]) / f["salt"]
    out["mirage_vs_salt_public"] = (f["salt-public"] - f["mirage"]) / f["salt-public"]
    out["mirage_vs_global"] = (f["global"] - f["mirage"]) / f["global"]
    return out
