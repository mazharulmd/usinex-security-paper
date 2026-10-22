# Pinned versions and testbed

## Testbed

**Not available yet.** The testbed (nodes, GPUs, NICs, RoCE/IB, SSD) has not been
provided. The development VM used so far (2 vCPU AMD EPYC 7J13, 3 GB RAM, no GPU,
no RDMA) runs only the client-side harness and its tests; no measurement comes from it.

## Serving stack (proposed, not installed)

| Component | Pin | Notes |
| --- | --- | --- |
| vLLM | v0.31.0 (2026-10-05) | Connector usage read from `docs/features/mooncake_store_connector_usage.md` at this tag |
| Mooncake (`mooncake-transfer-engine`) | v0.3.13.post1 (2026-08-31) | To verify at install: `standalone-store` mode and `enable_offload` as vLLM v0.31.0 expects |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, TP=1 | Gated on Hugging Face; needs access |

## Data

| Item | Pin |
| --- | --- |
| Mooncake traces | commit `5872ee870dad0a65fe3557786f66cd7d13405816`, `FAST25-release/traces/` (fetched and SHA-256-checked by `scripts/fetch_traces.sh`) |
| Prompt synthesis | `harness/synth.py`, SYNTH_VERSION 1 |

## Client-side Python (`requirements.txt`)

Python 3.12.3; aiohttp 3.14.5, matplotlib 3.11.2, numpy 2.5.4, pytest 9.1.1,
pytest-asyncio 1.4.0, scikit-learn 1.9.1.
