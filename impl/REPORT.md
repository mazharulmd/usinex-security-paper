# Mirage phase 1 report

Every number here links to a raw-log directory under `results/`. Nothing is
reported that was not measured.

## Status

**No testbed measurements.** There is no GPU testbed (2026-10-22). The only
completed experiment is the trace-driven slack simulation below, which needs
no serving runs.

## Week-1 gate (tier separability)

Not run.

## Week-2 gate (proxy residual AUC, deadline misses)

Not run.

## Week-3: slack from trace-driven simulation

**Run:** [`results/2026-10-22_slack-sim/`](results/2026-10-22_slack-sim/), code at
commit `9ba29d2` (clean tree). `summary.csv` is rebuilt from the per-request raw
files by `make slack-report RUN=results/2026-10-22_slack-sim` (checked: byte-identical).

**Method** (full rules in `sim/slack.py`). For each request, in trace order:
`p_glob` = prefix hit in one global LRU pool; `p_virt` = prefix hit in that
principal's own LRU (quota Q) or the declared-public set. Slack = prefill work
at `p_virt` minus at `p_glob`, in FLOPs from Qwen2.5-7B-Instruct's architecture
(not seconds: converting needs measured prefill throughput). Simplifications:
blocks are inserted at arrival (no in-flight overlap), only full 512-token blocks
are reused, public blocks are pre-computed and pinned. 144 configurations: 2
traces × 6 principal assignments × public set {none, popular} × pool
{unbounded, 2048 GB, 512 GB} × quota {unbounded, 1M tokens}.

**Principals.** The traces have no principal IDs. Sessions are found by linking
a request to the earlier request it shares the longest prefix with, if the
deepest shared block is used by ≤ 50 requests. Shared blocks are either very
popular (> 1000 requests: templates) or rare (≤ 50: history); none falls in
between, so the threshold is not a free choice. This gives 7,373 sessions
(conversation, 1 popular block) and 18,781 sessions (tool&agent, 18 popular
blocks). Sessions are then mapped to principals in several ways.

**Results**, unbounded pool and quota, so only cross-principal reuse counts.
Each cell: slack as a share of Mirage's budgeted prefill FLOPs / as a share of
what global sharing saves. Global sharing saves 35.6% (conversation) and 52.1%
(tool&agent) of prefill FLOPs versus no cache.

| Principal assignment | Public set | Conversation | Tool & agent |
| --- | --- | --- | --- |
| single principal (sanity check) | none | 0.0% / 0.0% | 0.0% / 0.0% |
| sessions → 10 principals | none | 0.0% / 0.0% | 0.1% / 0.1% |
| sessions → 100 principals | none | 0.0% / 0.1% | 0.7% / 0.7% |
| sessions → 1000 principals | none | 0.4% / 0.7% | 7.0% / 6.9% |
| each session its own principal | none | 2.6% / 4.9% | 35.5% / 50.5% |
| any session-based assignment | popular blocks | 0.0% / 0.0% | 0.0% / 0.0% |
| each request random of 100 (upper bound) | none | 33.1% / 89.5% | 33.2% / 45.7% |
| each request random of 100 (upper bound) | popular blocks | 33.1% / 89.4% | 32.9% / 45.0% |

**What this shows.**

1. **The traces cannot settle whether slack is material.** Under session-based
   assignment, all cross-principal reuse is reuse of the popular template blocks.
   It ranges from ~0% to 35.5% depending only on how many principals the sessions
   are spread over, and is exactly 0 once those templates are declared public.
   Splitting sessions across principals (the upper bound) gives ~33%, but that
   breaks every multi-turn conversation across tenants, which is unrealistic.
2. **The 0% with a public set is partly circular.** Session clustering puts all
   rare-block sharing inside one session, hence one principal. Cross-tenant
   sharing of non-template content cannot appear under it by construction.
3. **Slack also comes from the quota, not only from other principals.** A single
   principal with a 1M-token quota and an unbounded pool has 32.9% (conversation)
   and 32.5% (tool&agent) slack: its own evicted history is still physically
   cached. That slack is real, but it is paid for by giving the principal worse
   TTFT than global sharing on its own content.
4. **A finite pool creates virtual hits on evicted blocks (§5.6's pinning problem).**
   At 512 GB with sessions → 100 principals and a 1M-token quota, 988 of 12,031
   (8.2%, conversation) and 746 of 23,608 (3.2%, tool&agent) requests have
   `p_virt > p_glob`, i.e. would recompute and risk a deadline miss; net slack is
   negative (−13.3% and −8.2%). Sum of quotas (100M tokens ≈ 5.6 TB of KV) far
   exceeds the pool.
5. 828 (conversation) and 725 (tool&agent) requests exceed Qwen2.5-7B's
   native 32,768-token context; testbed runs will need YaRN or a length cap.

## Week-3 go/pivot

Not decided. The brief's criterion needs "slack is material", which has no
threshold yet, and A2–A4/A6 cannot be confirmed without a testbed. The slack
result above depends on an unknowable principal assignment; on the most
plausible readings (sessions belong to one tenant; shared templates are public
or one tenant's own) it is near zero.

## Observations from documentation and data (not measurements)

These come from reading the Mooncake and vLLM sources and the trace files, not from
running anything. They may affect the paper; the paper's .tex files are not edited here.

1. **Trace block size is 512 tokens.** The Mooncake FAST'25 trace README says so, and
   all 39,632 rows across the three traces satisfy `len(hash_ids) == ceil(input_length/512)`.
   The paper's §5.1 uses 256 tokens per block.
2. **Week-1 gate tests A1, not A2.** The gate uses miss-vs-hit AUC but its consequence
   is cutting A2, which needs tiers separable from each other. Proposed: gate A2 on
   which-tier AUC as well; report both.
3. **Mooncake multi-tenant mode may close A6.** Current Mooncake docs describe
   `tenant_id` namespaces with tenant-scoped eviction under quota (`--enable_multi_tenants`).
   If enabled, another tenant's inserts may not evict the attacker's blocks. It is set
   per deployment, not per request. It should be a baseline for A6.
4. **Soft pins are set at write time.** Mooncake's soft pin is a `ReplicateConfig`
   option on put (set by the connector inside vLLM) and reads do not extend it. The
   paper's §5.6 pins blocks from the scheduler; doing that without touching connector
   code (rule 3 / G4) needs checking.
5. **`cache_salt` likely salts store keys too.** In vLLM v0.31.0 it is folded into the
   first block hash, and store keys appear to be built from block hashes. To be
   confirmed on the testbed, since the Salt baseline depends on it.
