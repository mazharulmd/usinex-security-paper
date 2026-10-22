"""Synthesize token-ID prompts whose block structure matches a Mooncake trace.

Mooncake's released traces give each request a list of remapped prefix block
hashes (``hash_ids``), one per 512-token block; equal IDs mean reusable KV. We
need prompts with exactly that sharing structure:

  * same (namespace, hash_id)  -> byte-identical 512-token block
  * different hash_id          -> independent pseudo-random block

Specification (SYNTH_VERSION 1), so the prompts can be regenerated in any
language:

  For block key (namespace, seed, h), tokens are drawn from a SHA-256 counter
  stream. Counter c = 0, 1, 2, ... gives the 32-byte digest

      SHA-256( "mirage-synth-v1|" + namespace + "|" + str(seed) + "|" + str(h)
               + "|" + c as 8-byte big-endian )

  read as eight big-endian uint32 words w. With the vocabulary range [lo, hi)
  and n = hi - lo, words w >= 2^32 - (2^32 mod n) are rejected (no modulo bias);
  otherwise the next token is lo + (w mod n). The block is the first 512 tokens.

  A request's prompt is the concatenation of its blocks, truncated to
  ``input_length``. The traces satisfy len(hash_ids) == ceil(input_length/512)
  for every row (checked on load), so only the last block can be partial, and
  only full blocks can be shared.

Prompts are sent to vLLM as token-ID lists, never as text, so retokenization
cannot merge tokens across block boundaries. 512 is a multiple of vLLM's block
size (16), so trace blocks also land on vLLM block boundaries.

Namespaces keep independent ID spaces apart: each trace file uses its own
name (conversation and toolagent IDs both start at 0), and attacker or victim
probes use their own namespaces.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SYNTH_VERSION = 1
BLOCK_TOKENS = 512

# Llama-3 reserves IDs >= 128000 for special tokens; the low IDs are byte-level
# pieces. This range avoids both. It is a parameter and recorded in run metadata.
DEFAULT_VOCAB = (1000, 100000)


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 0
    vocab_lo: int = DEFAULT_VOCAB[0]
    vocab_hi: int = DEFAULT_VOCAB[1]
    block_tokens: int = BLOCK_TOKENS

    def as_dict(self) -> dict:
        return {"synth_version": SYNTH_VERSION, "seed": self.seed,
                "vocab_lo": self.vocab_lo, "vocab_hi": self.vocab_hi,
                "block_tokens": self.block_tokens}


@lru_cache(maxsize=65536)
def block_tokens(namespace: str, h: int, cfg: SynthConfig = SynthConfig()) -> tuple[int, ...]:
    """The fixed token block for hash ID ``h`` in ``namespace``."""
    n = cfg.vocab_hi - cfg.vocab_lo
    if n <= 0:
        raise ValueError("empty vocabulary range")
    limit = (1 << 32) - ((1 << 32) % n)
    head = f"mirage-synth-v1|{namespace}|{cfg.seed}|{h}|".encode()
    out: list[int] = []
    c = 0
    while len(out) < cfg.block_tokens:
        d = hashlib.sha256(head + c.to_bytes(8, "big")).digest()
        for i in range(0, 32, 4):
            w = int.from_bytes(d[i:i + 4], "big")
            if w < limit:
                out.append(cfg.vocab_lo + w % n)
                if len(out) == cfg.block_tokens:
                    break
        c += 1
    return tuple(out)


def prompt_tokens(namespace: str, hash_ids: list[int], input_length: int,
                  cfg: SynthConfig = SynthConfig()) -> list[int]:
    """Token-ID prompt for one trace row."""
    if len(hash_ids) != math.ceil(input_length / cfg.block_tokens):
        raise ValueError(f"{len(hash_ids)} hash_ids for {input_length} tokens "
                         f"(block size {cfg.block_tokens})")
    toks: list[int] = []
    for h in hash_ids:
        toks.extend(block_tokens(namespace, h, cfg))
    return toks[:input_length]


@dataclass(frozen=True)
class TraceRow:
    index: int
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]


def load_trace(path: str | Path, block: int = BLOCK_TOKENS) -> tuple[list[TraceRow], str]:
    """Load a Mooncake trace; return rows and the file's SHA-256."""
    raw = Path(path).read_bytes()
    rows = []
    for i, line in enumerate(raw.decode().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        row = TraceRow(i, int(r["timestamp"]), int(r["input_length"]),
                       int(r["output_length"]), tuple(r["hash_ids"]))
        if len(row.hash_ids) != math.ceil(row.input_length / block):
            raise ValueError(f"{path}:{i + 1}: hash_ids do not match input_length")
        rows.append(row)
    return rows, hashlib.sha256(raw).hexdigest()


def trace_namespace(path: str | Path) -> str:
    """Namespace for a trace file: its stem, e.g. 'conversation_trace'."""
    return "trace:" + Path(path).stem
