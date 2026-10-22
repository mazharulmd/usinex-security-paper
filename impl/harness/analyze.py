"""Per-tier TTFT distributions and tier-classifier AUCs from raw sweep records.

Input: one or more ``requests.jsonl`` files from a tier sweep. Probe records
carry labels ``tier`` (one of TIERS), ``load`` and ``prompt_len``; records with
``labels.role == "background"`` are ignored. A probe whose ground-truth tier
label could not be confirmed carries ``labels.label_ok = false`` and is dropped
(and counted).

For each (load, prompt_len) group:

  miss_vs_hit  binary: miss vs. any hit tier                      (attack A1)
  which_tier   multiclass over the hit tiers only, macro one-vs-rest AUC  (A2)

Each is a classifier on log(TTFT) evaluated by stratified k-fold
cross-validation: the AUC is computed on out-of-fold scores, so the classifier
never scores probes it was trained on. A 95% CI comes from bootstrapping the
out-of-fold scores. Outputs, written next to the inputs or to --out:

  probes.csv         one row per kept probe
  tier_stats.csv     n, mean, p50, p90, p99 TTFT per (load, prompt_len, tier)
  auc.csv            AUC, CI, accuracy, n per (load, prompt_len, task)
  gate.json          week-1 gate outcome at the gate load level
  tier_ladder.png    TTFT CDFs per tier, one panel per load level
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .client import read_records

TIERS = ["miss", "hbm", "local_dram", "remote_dram", "ssd"]
HIT_TIERS = TIERS[1:]
TIER_NAMES = {"miss": "Miss", "hbm": "Local HBM", "local_dram": "Local DRAM",
              "remote_dram": "Remote DRAM", "ssd": "SSD"}
LOADS = ["idle", "moderate", "near_slo"]


def extract(paths: list[Path]) -> tuple[list[dict], Counter]:
    """Probe rows with a TTFT, and counts of what was dropped and why."""
    rows, dropped = [], Counter()
    for p in paths:
        for r in read_records(p):
            lab = r.get("labels") or {}
            if lab.get("role") == "background":
                continue
            if lab.get("tier") not in TIERS:
                dropped["no_tier_label"] += 1
            elif lab.get("label_ok") is False:
                dropped["tier_label_unconfirmed"] += 1
            elif r.get("http_status") != 200:
                dropped[f"http_{r.get('http_status')}"] += 1
            elif r.get("ttft_ns") is None:
                dropped["no_first_token"] += 1
            else:
                rows.append({"req_id": r["req_id"], "tier": lab["tier"],
                             "load": lab.get("load", "unknown"),
                             "prompt_len": int(lab.get("prompt_len", r["prompt_len"])),
                             "ttft_ms": r["ttft_ns"] / 1e6, "source": str(p)})
    return rows, dropped


def make_classifier(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=1000)
    if kind == "gbdt":
        return HistGradientBoostingClassifier(max_iter=100, random_state=0)
    raise ValueError(kind)


def oof_scores(x: np.ndarray, y: np.ndarray, kind: str, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold class probabilities, shape (n, n_classes)."""
    classes = np.unique(y)
    out = np.zeros((len(y), len(classes)))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    X = np.log(x).reshape(-1, 1)
    for tr, te in skf.split(X, y):
        clf = make_classifier(kind).fit(X[tr], y[tr])
        idx = np.searchsorted(classes, clf.classes_)
        out[np.ix_(te, idx)] = clf.predict_proba(X[te])
    return out


def auc_of(y: np.ndarray, scores: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], scores[:, 1]))
    return float(roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=classes))


def evaluate(x: np.ndarray, y: np.ndarray, kind: str = "gbdt", folds: int = 5,
             boot: int = 1000, seed: int = 0) -> dict | None:
    counts = Counter(y.tolist())
    if len(counts) < 2 or min(counts.values()) < folds:
        return None
    scores = oof_scores(x, y, kind, folds, seed)
    auc = auc_of(y, scores)
    classes = np.unique(y)
    acc = float(np.mean(classes[scores.argmax(1)] == y))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == len(classes):
            bs.append(auc_of(y[i], scores[i]))
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return {"auc": auc, "ci_lo": float(lo), "ci_hi": float(hi), "accuracy": acc,
            "n": int(len(y)), "classes": "|".join(map(str, classes)),
            "per_class_n": "|".join(f"{c}:{counts[c]}" for c in classes)}


def analyze(rows: list[dict], kind: str, folds: int, boot: int, seed: int) -> tuple[list, list]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["load"], r["prompt_len"])].append(r)
    stats, aucs = [], []
    for (load, plen), g in sorted(groups.items()):
        by_tier = defaultdict(list)
        for r in g:
            by_tier[r["tier"]].append(r["ttft_ms"])
        for t in TIERS:
            v = np.array(by_tier.get(t, []))
            if len(v):
                stats.append({"load": load, "prompt_len": plen, "tier": t, "n": len(v),
                              "mean_ms": v.mean(), "p50_ms": np.percentile(v, 50),
                              "p90_ms": np.percentile(v, 90), "p99_ms": np.percentile(v, 99)})
        x = np.array([r["ttft_ms"] for r in g])
        tier = np.array([r["tier"] for r in g])
        tasks = {"miss_vs_hit": (x, np.where(tier == "miss", "miss", "hit")),
                 "which_tier": (x[tier != "miss"], tier[tier != "miss"])}
        for task, (xt, yt) in tasks.items():
            res = evaluate(xt, yt, kind, folds, boot, seed)
            row = {"load": load, "prompt_len": plen, "task": task, "classifier": kind,
                   "folds": folds}
            row.update(res or {"auc": None, "note": "too few probes per class"})
            aucs.append(row)
    return stats, aucs


def gate(aucs: list[dict], load: str, threshold: float) -> dict:
    """Week-1 gate. A1 passes on miss_vs_hit AUC; A2 (proposed split) on which_tier AUC."""
    out = {"load": load, "threshold": threshold, "by_prompt_len": {}}
    for row in aucs:
        if row["load"] != load:
            continue
        d = out["by_prompt_len"].setdefault(str(row["prompt_len"]), {})
        d[row["task"]] = {"auc": row["auc"], "ci_lo": row.get("ci_lo"),
                          "pass": row["auc"] is not None and row["auc"] > threshold}
    results = out["by_prompt_len"].values()
    for task, name in (("miss_vs_hit", "A1"), ("which_tier", "A2")):
        verdicts = [d[task]["pass"] for d in results if task in d]
        out[name] = ("not_measured" if not verdicts
                     else "pass" if any(verdicts) else "fail")
    return out


def plot(rows: list[dict], path: Path, prompt_len: int | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    # Reference categorical palette, light mode, fixed order (slots 1-5).
    colors = dict(zip(TIERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e0"
    if prompt_len is None:
        prompt_len = Counter(r["prompt_len"] for r in rows).most_common(1)[0][0]
    sel = [r for r in rows if r["prompt_len"] == prompt_len]
    loads = [l for l in LOADS if any(r["load"] == l for r in sel)] or \
        sorted({r["load"] for r in sel})
    fig, axes = plt.subplots(1, len(loads), figsize=(3.4 * len(loads), 3.0),
                             sharey=True, squeeze=False)
    for ax, load in zip(axes[0], loads):
        for t in TIERS:
            v = np.sort([r["ttft_ms"] for r in sel if r["load"] == load and r["tier"] == t])
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=colors[t], lw=2,
                        drawstyle="steps-post", label=TIER_NAMES[t])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(load.replace("_", "-"), color=ink, fontsize=10)
        ax.set_xlabel("TTFT (ms, log scale)", color=muted, fontsize=9)
        ax.grid(True, color=grid, lw=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for s in ax.spines.values():
            s.set_color(grid)
    axes[0][0].set_ylabel("Fraction of probes", color=muted, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               frameon=False, labelcolor=ink)
    fig.suptitle(f"TTFT by serving tier, {prompt_len}-token prefix", color=ink, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Analyze a tier sweep from raw records.")
    ap.add_argument("inputs", nargs="+", type=Path, help="requests.jsonl files or run dirs")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--classifier", choices=["gbdt", "logreg"], default="gbdt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-load", default="moderate")
    ap.add_argument("--gate-threshold", type=float, default=0.8)
    a = ap.parse_args(argv)
    files = [p / "requests.jsonl" if p.is_dir() else p for p in a.inputs]
    out = a.out or files[0].parent
    out.mkdir(parents=True, exist_ok=True)

    rows, dropped = extract(files)
    if not rows:
        raise SystemExit(f"no usable probes in {files}; dropped: {dict(dropped)}")
    stats, aucs = analyze(rows, a.classifier, a.folds, a.bootstrap, a.seed)
    write_csv(out / "probes.csv", rows)
    write_csv(out / "tier_stats.csv", stats)
    write_csv(out / "auc.csv", aucs)
    g = gate(aucs, a.gate_load, a.gate_threshold)
    g.update({"inputs": [str(f) for f in files], "dropped": dict(dropped),
              "analysis_argv": sys.argv if argv is None else argv})
    (out / "gate.json").write_text(json.dumps(g, indent=2) + "\n")
    plot(rows, out / "tier_ladder.png")
    return out


if __name__ == "__main__":
    print(main())
