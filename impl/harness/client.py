"""Streaming OpenAI-compatible client that records per-request timing.

Each request is a POST to /v1/completions with the prompt as a token-ID list
(see synth.py), ``stream=true`` and ``stream_options.include_usage=true``. For
every request we record, on the client's monotonic clock (``time.monotonic_ns``):

  t_send_ns        just before the request is written
  t_headers_ns     response status and headers received
  t_first_token_ns first SSE chunk that carries a choice
  token_times_ns   arrival of every SSE chunk that carries a choice
  t_done_ns        stream finished (or failed)

plus the HTTP status, all response headers, the full final ``usage`` object
(including ``prompt_tokens_details.cached_tokens`` when the server reports it)
and any error. ``t_send_wall_ns`` (``time.time_ns``) is kept only to line
records up with server logs; intervals always use the monotonic clock.

One SSE chunk is treated as one token event. vLLM streams one chunk per
generated token, but it can buffer partial UTF-8 into an empty-text chunk, so
we count chunks with a choice regardless of their text.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import aiohttp

MAX_ERROR_BODY = 2000


@dataclass
class Request:
    req_id: str
    prompt: list[int]
    max_tokens: int
    principal: str = "default"           # sent as the API key (Bearer token)
    cache_salt: str | None = None        # vLLM per-request salt (Salt baseline)
    labels: dict[str, Any] = field(default_factory=dict)  # tier, load, role, ...
    scheduled_ns: int | None = None      # intended send time, for open-loop lag


@dataclass
class Record:
    req_id: str
    principal: str
    url: str
    prompt_len: int
    max_tokens: int
    labels: dict[str, Any]
    scheduled_ns: int | None = None
    t_send_ns: int | None = None
    t_send_wall_ns: int | None = None
    t_headers_ns: int | None = None
    t_first_token_ns: int | None = None
    t_done_ns: int | None = None
    token_times_ns: list[int] = field(default_factory=list)
    http_status: int | None = None
    headers: dict[str, str] = field(default_factory=dict)
    usage: dict | None = None
    finish_reason: str | None = None
    error: str | None = None

    @property
    def ttft_ns(self) -> int | None:
        if self.t_first_token_ns is None or self.t_send_ns is None:
            return None
        return self.t_first_token_ns - self.t_send_ns

    def to_json(self) -> str:
        d = asdict(self)
        d["ttft_ns"] = self.ttft_ns
        return json.dumps(d, separators=(",", ":"))


class Client:
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 600.0,
                 ignore_eos: bool = True, extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.ignore_eos = ignore_eos     # fixed output length regardless of content
        self.extra_body = extra_body or {}
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Client":
        # No connection cap: client-side queueing would show up as server latency.
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0), timeout=self._timeout)
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._session is not None
        await self._session.close()

    def body(self, req: Request) -> dict:
        b = {"model": self.model, "prompt": req.prompt, "max_tokens": req.max_tokens,
             "temperature": 0.0, "stream": True,
             "stream_options": {"include_usage": True}, **self.extra_body}
        if self.ignore_eos:
            b["ignore_eos"] = True
        if req.cache_salt is not None:
            b["cache_salt"] = req.cache_salt
        return b

    async def send(self, req: Request) -> Record:
        assert self._session is not None, "use 'async with Client(...)'"
        rec = Record(req.req_id, req.principal, self.url, len(req.prompt),
                     req.max_tokens, req.labels, req.scheduled_ns)
        payload = json.dumps(self.body(req))
        headers = {"Authorization": f"Bearer {req.principal}",
                   "Content-Type": "application/json"}
        try:
            rec.t_send_wall_ns = time.time_ns()
            rec.t_send_ns = time.monotonic_ns()
            async with self._session.post(self.url, data=payload, headers=headers) as resp:
                rec.t_headers_ns = time.monotonic_ns()
                rec.http_status = resp.status
                rec.headers = dict(resp.headers)
                if resp.status != 200:
                    rec.error = (await resp.text())[:MAX_ERROR_BODY]
                    return rec
                await self._read_stream(resp, rec)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as e:
            rec.error = f"{type(e).__name__}: {e}"
        finally:
            rec.t_done_ns = time.monotonic_ns()
        return rec

    @staticmethod
    async def _read_stream(resp: aiohttp.ClientResponse, rec: Record) -> None:
        while True:
            line = await resp.content.readline()
            now = time.monotonic_ns()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                rec.usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                if rec.t_first_token_ns is None:
                    rec.t_first_token_ns = now
                rec.token_times_ns.append(now)
                if choices[0].get("finish_reason"):
                    rec.finish_reason = choices[0]["finish_reason"]


class RecordWriter:
    """Append-only JSONL writer; one line per completed request, flushed."""

    def __init__(self, path: Path):
        self.path = path
        self._f: IO[str] = open(path, "a", buffering=1)

    def write(self, rec: Record) -> None:
        self._f.write(rec.to_json() + "\n")

    def close(self) -> None:
        self._f.close()


def read_records(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
