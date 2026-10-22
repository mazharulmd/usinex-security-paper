# Mirage phase 1 report

Every number here links to a raw-log directory under `results/`. Nothing is
reported that was not measured.

## Status

**No measurements yet.** The serving testbed is not available; only the
client-side harness exists (see `README.md`).

## Week-1 gate (tier separability)

Not run.

## Week-2 gate (proxy residual AUC, deadline misses)

Not run.

## Week-3 go/pivot

Not decided.

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
