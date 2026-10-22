"""A mock OpenAI-compatible streaming server for testing the harness without GPUs.

It imitates the parts of vLLM the harness depends on: /v1/completions with a
token-ID prompt, SSE streaming, a final usage chunk with
``prompt_tokens_details.cached_tokens``, ``cache_salt``, and HTTP 429. Its
timing is a made-up model (prefix-cache hit/miss with optional jitter) with
nothing to do with real hardware. **Nothing measured against it is a result.**
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import OrderedDict
from dataclasses import dataclass

from aiohttp import web

PROMPT_TOKENS_DETAILS = web.AppKey("prompt_tokens_details", bool)


@dataclass
class MockConfig:
    block: int = 16
    capacity_blocks: int = 100_000
    base_ms: float = 5.0
    miss_ms_per_token: float = 0.05
    hit_ms_per_token: float = 0.005
    tbt_ms: float = 2.0
    jitter_ms: float = 0.0
    max_inflight: int | None = None      # reply 429 beyond this many concurrent requests
    seed: int = 0


class MockServer:
    def __init__(self, cfg: MockConfig = MockConfig()):
        self.cfg = cfg
        self.cache: OrderedDict[bytes, None] = OrderedDict()
        self.inflight = 0
        self.rng = random.Random(cfg.seed)

    def _block_hashes(self, prompt: list[int], salt: str | None) -> list[bytes]:
        out, prev = [], (salt or "").encode()
        for i in range(0, len(prompt) - len(prompt) % self.cfg.block, self.cfg.block):
            prev = hashlib.sha256(prev + json.dumps(prompt[i:i + self.cfg.block]).encode()).digest()
            out.append(prev)
        return out

    def lookup_and_insert(self, prompt: list[int], salt: str | None) -> int:
        hashes = self._block_hashes(prompt, salt)
        hit = 0
        for h in hashes:
            if h not in self.cache:
                break
            hit += 1
        for h in hashes:
            self.cache[h] = None
            self.cache.move_to_end(h)
        while len(self.cache) > self.cfg.capacity_blocks:
            self.cache.popitem(last=False)
        return hit * self.cfg.block

    async def completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt = body["prompt"]
        if not (isinstance(prompt, list) and all(isinstance(t, int) for t in prompt)):
            return web.json_response({"error": "mock accepts token-ID prompts only"}, status=400)
        if self.cfg.max_inflight is not None and self.inflight >= self.cfg.max_inflight:
            return web.json_response({"error": "too many requests"}, status=429)
        self.inflight += 1
        try:
            cached = self.lookup_and_insert(prompt, body.get("cache_salt"))
            n, m = len(prompt), int(body.get("max_tokens", 16))
            c = self.cfg
            ttft = c.base_ms + (n - cached) * c.miss_ms_per_token + cached * c.hit_ms_per_token
            ttft += abs(self.rng.gauss(0, c.jitter_ms)) if c.jitter_ms else 0.0

            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await asyncio.sleep(ttft / 1000)
            for i in range(m):
                if i:
                    await asyncio.sleep(c.tbt_ms / 1000)
                choice = {"index": 0, "text": f" t{i}",
                          "finish_reason": "length" if i == m - 1 else None}
                await resp.write(b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n")
            usage = {"prompt_tokens": n, "completion_tokens": m, "total_tokens": n + m}
            if request.app[PROMPT_TOKENS_DETAILS]:
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            await resp.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            self.inflight -= 1


def make_app(cfg: MockConfig = MockConfig(), prompt_tokens_details: bool = True) -> web.Application:
    srv = MockServer(cfg)
    app = web.Application()
    app[PROMPT_TOKENS_DETAILS] = prompt_tokens_details
    app.router.add_post("/v1/completions", srv.completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--max-inflight", type=int)
    a = ap.parse_args()
    web.run_app(make_app(MockConfig(jitter_ms=a.jitter_ms, max_inflight=a.max_inflight)),
                port=a.port)


if __name__ == "__main__":
    main()
