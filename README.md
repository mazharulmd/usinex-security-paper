# Mirage — USENIX Security draft (first full skeleton, 2026-10-22)

**Mirage: Physically Shared, Observably Private KVCache for Disaggregated LLM Serving**

## Build

```
latexmk -pdf main.tex
```

Overleaf: upload this folder into a project made from the official USENIX template.
`main.tex` loads `usenix-2020-09.sty` automatically when it exists; otherwise it falls back
to `usenix-shim.sty`, a layout approximation. **Do not submit with the shim.**

## Placeholder legend (set `\draftfalse` in `macros.tex` to hide them)

| Macro | Colour | Meaning |
| --- | --- | --- |
| `\tbd{...}` | red bold | A number or result still to be measured |
| `\todo{...}` | blue | Writing or a decision still to do |
| `\needcite{...}` | orange | A claim that needs a source |
| `\placeholderfig{h}{...}` | dashed red box | A figure to be plotted from data |

## Status of claims

- **No result in this draft is measured.** Every quantitative claim about Mirage or the
  attacks is a `\tbd{}` target.
- Proposition 1 is a proof *sketch*; Conjecture 1 is unproved (Section 6).
- Facts about Mooncake come from the FAST '25 paper and the public repository.
- Bibliography: entries marked `VERIFY` in their `note` field still need checking
  (full first names, pages or final venue). Initials are used where full names were not
  confirmed. Run a citation check before submission.

## Layout

```
main.tex, macros.tex, references.bib, usenix-shim.sty
sections/00-abstract … 11-conclusion, A-ethics, B-open-science
```

## Next (weeks 1–3, from the research brief)

1. Stand up 2-node vLLM + Mooncake Store; measure miss vs. per-tier hit TTFT at three loads.
2. Proxy-level release gate; run A1, A2, A5 against it; measure residual AUC and miss rate.
3. Trace-driven slack estimate; testbed checks of A4 and A6. Go/pivot gate at end of week 3.
