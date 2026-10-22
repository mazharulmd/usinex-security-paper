"""Trace-driven estimate of Mirage's slack: prefill work saved by cross-principal hits.

For every request in a Mooncake trace we compute two prefix-hit lengths:

  p_glob  longest prefix of the request's blocks in the global physical cache
          (one shared LRU pool, what Mooncake reuses)
  p_virt  longest prefix in the principal's own virtual cache (an LRU of quota
          Q) or the declared-public set: what Mirage lets the principal observe

and the prefill work at each. Mirage budgets the deadline for p_virt but the
GPU only computes from p_glob, so the slack is  work(p_virt) - work(p_glob).

Work is counted in FLOPs from the model's architecture, not in seconds:
converting to GPU time needs measured prefill throughput, which needs the
testbed. Prefill of tokens [p, n) costs

  2 * W * (n - p)  +  4 * L * H * sum_{i=p}^{n-1} (i + 1)

where W is the number of weights in the transformer layers' linear maps,
L the layer count and H = heads * head_dim (QK^T and AV under a causal mask).
Norms, biases, rotary embeddings and the LM head (run on the last token only)
are left out.

Simulation rules, all simplifications to keep in mind when reading results:

  * Requests are processed in trace order and their blocks are inserted at
    arrival, as if prefill were instantaneous. Two near-simultaneous requests
    can therefore hit each other's blocks earlier than they would in reality.
  * Only full 512-token blocks are cached or reused; the trailing partial
    block is always recomputed. A full hit still recomputes the last token.
  * Caches are LRU over blocks; a lookup refreshes the blocks it hits.
  * Public blocks are treated as present in every virtual cache and are not
    charged to a principal's quota. The operator pre-computes and pins them,
    so they are also always present physically, outside the LRU capacity.

The traces carry no principal IDs, so principals are assigned (see ASSIGNMENTS);
the result depends heavily on that choice.
"""

from __future__ import annotations

import json
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from harness.synth import BLOCK_TOKENS, TraceRow


@dataclass(frozen=True)
class ModelCost:
    layers: int
    attn_width: int          # heads * head_dim
    linear_weights: int      # weights in all layers' q, k, v, o and MLP projections
    kv_bytes_per_token: int  # bf16 K and V across all layers

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "ModelCost":
        c = json.loads(Path(path).read_text())
        h, heads, kvh = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        head_dim = c.get("head_dim", h // heads)
        q, kv = heads * head_dim, kvh * head_dim
        per_layer = h * q + 2 * h * kv + q * h + 3 * h * c["intermediate_size"]
        return cls(c["num_hidden_layers"], q, c["num_hidden_layers"] * per_layer,
                   2 * c["num_hidden_layers"] * kv * 2)

    def prefill_flops(self, n: int, cached: int) -> int:
        p = min(cached, n - 1)                       # the last token is always computed
        causal = (n * (n + 1) - p * (p + 1)) // 2    # sum of (i + 1) for i in [p, n)
        return 2 * self.linear_weights * (n - p) + 4 * self.layers * self.attn_width * causal


class LRU:
    """LRU set of block IDs; capacity None means unbounded."""

    def __init__(self, capacity: int | None):
        self.capacity = capacity
        self.d: OrderedDict[int, None] = OrderedDict()

    def __contains__(self, h: int) -> bool:
        return h in self.d

    def touch(self, blocks) -> None:
        for h in blocks:
            self.d[h] = None
            self.d.move_to_end(h)
        if self.capacity is not None:
            while len(self.d) > self.capacity:
                self.d.popitem(last=False)


def prefix_hits(chain: tuple[int, ...], present) -> int:
    k = 0
    while k < len(chain) and present(chain[k]):
        k += 1
    return k


# ---------------------------------------------------------------- principals

def block_popularity(rows: list[TraceRow]) -> Counter:
    """Number of requests containing each block."""
    return Counter(h for r in rows for h in set(r.hash_ids))


def sessions(rows: list[TraceRow], pop: Counter, max_pop: int) -> list[int]:
    """Session ID per request.

    A request joins the session of the earlier request it shares the longest
    prefix with, if the deepest shared block is used by at most ``max_pop``
    requests. Sharing only popular blocks (templates, system prompts) does not
    link requests. Popularity can only fall along a chain, so the deepest shared
    block is the least popular one. In both real traces no deepest-shared block
    has popularity between 51 and 1000, so any max_pop in [50, 1000] gives the
    same sessions.
    """
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    last_with_prefix: dict[tuple[int, ...], int] = {}
    for i, r in enumerate(rows):
        c, k = r.hash_ids, 0
        while k < len(c) and c[:k + 1] in last_with_prefix:
            k += 1
        if k and pop[c[k - 1]] <= max_pop:
            parent[find(i)] = find(last_with_prefix[c[:k]])
        for j in range(1, len(c) + 1):
            last_with_prefix[c[:j]] = i
    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(len(rows))]


def assign(method: str, rows: list[TraceRow], sess: list[int], seed: int) -> list[int]:
    """Principal per request.

    single           everyone is one principal (sanity check: slack must be 0)
    session-each     each session is its own principal
    session-uniform:N  sessions spread uniformly at random over N principals
    request-random:N   each request independently uniform over N principals;
                       ignores sessions, so it is an upper bound on cross-principal reuse
    """
    rng = random.Random(seed)
    name, _, arg = method.partition(":")
    if name == "single":
        return [0] * len(rows)
    if name == "session-each":
        return list(sess)
    if name == "session-uniform":
        n = int(arg)
        mapping: dict[int, int] = {}
        return [mapping.setdefault(s, rng.randrange(n)) for s in sess]
    if name == "request-random":
        n = int(arg)
        return [rng.randrange(n) for _ in rows]
    raise ValueError(method)


# ---------------------------------------------------------------- simulation

@dataclass
class Outcome:
    principal: list[int]
    n: list[int]
    p_glob: list[int]   # blocks
    p_virt: list[int]   # blocks


def simulate(rows: list[TraceRow], principal: list[int], capacity_blocks: int | None,
             quota_blocks: int | None, public: frozenset[int]) -> Outcome:
    glob = LRU(capacity_blocks)
    vcd: dict[int, LRU] = {}
    out = Outcome([], [], [], [])
    for r, p in zip(rows, principal):
        chain = r.hash_ids[: r.input_length // BLOCK_TOKENS]   # full blocks only
        v = vcd.setdefault(p, LRU(quota_blocks))
        pg = prefix_hits(chain, lambda h: h in public or h in glob)
        pv = prefix_hits(chain, lambda h: h in public or h in v)
        glob.touch(h for h in chain if h not in public)
        v.touch(h for h in chain if h not in public)
        out.principal.append(p)
        out.n.append(r.input_length)
        out.p_glob.append(pg)
        out.p_virt.append(pv)
    return out


def summarize(o: Outcome, cost: ModelCost) -> dict:
    f0 = fg = fv = slack_pos = phantom = 0
    n_slack = n_phantom = tok_g = tok_v = 0
    for n, pg, pv in zip(o.n, o.p_glob, o.p_virt):
        cg, cv = min(pg * BLOCK_TOKENS, n - 1), min(pv * BLOCK_TOKENS, n - 1)
        a, g, v = cost.prefill_flops(n, 0), cost.prefill_flops(n, cg), cost.prefill_flops(n, cv)
        f0, fg, fv, tok_g, tok_v = f0 + a, fg + g, fv + v, tok_g + cg, tok_v + cv
        if v > g:
            slack_pos += v - g
            n_slack += 1
        elif g > v:
            phantom += g - v       # virtual says hit, physical copy was evicted
            n_phantom += 1
    saved_global = f0 - fg
    return {
        "requests": len(o.n), "principals": len(set(o.principal)),
        "prompt_tokens": sum(o.n), "cached_tokens_global": tok_g, "cached_tokens_virtual": tok_v,
        "flops_nocache": f0, "flops_global": fg, "flops_virtual": fv,
        "slack_flops": fv - fg, "slack_flops_positive": slack_pos, "phantom_flops": phantom,
        "requests_with_slack": n_slack, "requests_virtual_hit_physical_miss": n_phantom,
        "global_saving_frac": saved_global / f0 if f0 else 0.0,
        "slack_frac_of_virtual": (fv - fg) / fv if fv else 0.0,
        "slack_frac_of_global_saving": (fv - fg) / saved_global if saved_global else 0.0,
    }
