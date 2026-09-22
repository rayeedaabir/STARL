# =============================================================================
# STARL — significance tests + bootstrap confidence intervals
# =============================================================================
# NO GPU, NO torch. Runs in seconds. Two independent parts:
#
#   PART A (works now)  — paired non-parametric tests comparing rehearsal against
#                         each baseline, using the multi-seed results. Answers the
#                         reviewer question "is the difference statistically real?"
#   PART B (optional)   — bootstrap confidence intervals for any metric, computed by
#                         resampling the frozen test set. Needs per-image predictions,
#                         so it only runs if a predictions CSV is present; otherwise
#                         it is skipped with a note.
#
# RUN: Kaggle notebook, Accelerator = None, attach the aggregation output and both
# multiseed outputs. Or run locally with the CSVs in the working directory.

import os, glob, itertools, json
import numpy as np, pandas as pd

def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) or glob.glob(f"**/{pattern}", recursive=True)
    return sorted(set(hits))

try:
    from scipy.stats import wilcoxon, ttest_rel
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[note] scipy unavailable — falling back to an exact sign test")

def sign_test(d):
    """Exact two-sided sign test; used when scipy is absent."""
    from math import comb
    pos = int(np.sum(np.array(d) > 0)); n = int(np.sum(np.array(d) != 0))
    if n == 0: return 1.0
    tail = sum(comb(n, k) for k in range(min(pos, n - pos) + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def paired_report(name, a, b, labels):
    """a, b are paired arrays (rehearsal vs baseline)."""
    d = np.asarray(a) - np.asarray(b)
    out = {"comparison": name, "n_pairs": len(d), "mean_diff": float(d.mean()),
           "median_diff": float(np.median(d)), "min_diff": float(d.min()),
           "max_diff": float(d.max()), "n_favouring_rehearsal": int((d > 0).sum())}
    if HAVE_SCIPY and len(d) >= 5:
        try:
            w = wilcoxon(d); out["wilcoxon_p"] = float(w.pvalue)
        except Exception as e:
            out["wilcoxon_p"] = float("nan"); print(f"  [warn] wilcoxon {name}: {e}")
        try:
            t = ttest_rel(a, b); out["ttest_p"] = float(t.pvalue)
        except Exception:
            out["ttest_p"] = float("nan")
    else:
        out["sign_test_p"] = sign_test(d)
    # Cohen's d for paired samples
    out["cohens_d"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")
    print(f"\n  {name}")
    print(f"    n = {out['n_pairs']}, favouring rehearsal in {out['n_favouring_rehearsal']}/{out['n_pairs']}")
    print(f"    mean difference {out['mean_diff']:+.4f}  (range {out['min_diff']:+.4f} to {out['max_diff']:+.4f})")
    for k in ("wilcoxon_p", "ttest_p", "sign_test_p"):
        if k in out: print(f"    {k} = {out[k]:.2e}")
    print(f"    Cohen's d (paired) = {out['cohens_d']:.2f}")
    return out


# ============================ PART A ========================================
print("=" * 72); print("PART A — paired tests, rehearsal vs each baseline"); print("=" * 72)
rows = []

# ---- A1: rehearsal vs naive, across (model, seed) from the multiseed runs ----
ms = [pd.read_csv(f) for f in find("clmetrics_seed*.csv")]
if ms:
    ms = pd.concat(ms, ignore_index=True)
    piv = ms.pivot_table(index=["seed", "model"], columns="strategy", values="ACC")
    piv = piv.dropna(subset=["naive", "rehearsal"])
    print(f"\nmulti-seed pairs found: {len(piv)}")
    rows.append(paired_report("rehearsal vs naive (multi-seed, per model×seed)",
                              piv["rehearsal"].values, piv["naive"].values, piv.index))
else:
    print("\n[skip] no clmetrics_seed*.csv found — attach the multiseed outputs")

# ---- A2: rehearsal vs naive / LwF / EWC across the twelve architectures ----
cl = find("cl_metrics_by_strategy.csv")
if cl:
    raw = pd.read_csv(cl[0], header=[0, 1])
    raw.columns = ["|".join(str(x) for x in c).strip() for c in raw.columns]
    mcol = [c for c in raw.columns if c.lower().startswith("model")][0]
    def col(metric, strat):
        hit = [c for c in raw.columns if c.startswith(metric + "|") and c.endswith("|" + strat)]
        return raw[hit[0]].astype(float).values if hit else None
    reh = col("ACC", "rehearsal")
    if reh is not None:
        for strat in ("naive", "lwf", "ewc"):
            v = col("ACC", strat)
            if v is not None:
                rows.append(paired_report(f"rehearsal vs {strat} (12 architectures, seed 42)", reh, v, raw[mcol]))
        # and the non-replay methods against naive — the key negative result
        nv = col("ACC", "naive")
        for strat in ("lwf", "ewc"):
            v = col("ACC", strat)
            if v is not None and nv is not None:
                d = v - nv
                print(f"\n  {strat} vs naive (is the non-replay method any better?)")
                print(f"    mean difference {d.mean():+.4f}; better in {(d>0).sum()}/{len(d)} architectures")
                p = wilcoxon(d).pvalue if (HAVE_SCIPY and len(d) >= 5) else sign_test(d)
                print(f"    p = {p:.3f}  ->  {'no evidence of a difference' if p > 0.05 else 'difference detected'}")
                rows.append({"comparison": f"{strat} vs naive", "n_pairs": int(len(d)),
                             "mean_diff": float(d.mean()), "n_favouring_first": int((d > 0).sum()),
                             "wilcoxon_p" if HAVE_SCIPY else "sign_test_p": float(p)})
else:
    print("\n[skip] cl_metrics_by_strategy.csv not found")

if rows:
    out = pd.DataFrame(rows)
    out.to_csv("significance_tests.csv", index=False)
    print("\nwrote significance_tests.csv")
    print("\n=== SUMMARY (paste into the paper) ===")
    print(out.to_string(index=False))

print("""
HOW TO REPORT THIS
  With seven models x two extra seeds the Wilcoxon signed-rank test on fourteen
  paired differences reaches its minimum attainable p-value when every pair points
  the same way, which is what we observe. Report the test, the number of pairs, the
  direction of every pair and the effect size together; a p-value alone from a
  sample this small is weak evidence, whereas "14/14 pairs, mean +0.49, d > 3" is
  not. For LwF and EWC versus naive the correct conclusion is the absence of a
  detectable difference, which is a finding, not a failed test.
""")

# ============================ PART B ========================================
print("=" * 72); print("PART B — bootstrap confidence intervals"); print("=" * 72)
print("""
WHAT THIS IS, IN PLAIN TERMS
  A single-seed model gives one accuracy number with no error bar. Bootstrapping
  produces one without retraining anything. The frozen test set is a sample of the
  patients the model might have seen; if a slightly different sample had been drawn,
  the accuracy would have come out slightly differently. We simulate that by drawing,
  with replacement, a new test set of the same size from the one we have, recomputing
  the metric, and repeating a few thousand times. The middle 95 % of those values is
  the confidence interval: it expresses uncertainty due to the finite test set.
  Note this is NOT the same as seed variance, which expresses uncertainty due to
  training; ideally the paper reports both, and they answer different questions.
""")

def bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=2000, seed=42, alpha=0.05):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    fn = {"accuracy": accuracy_score,
          "balanced_accuracy": balanced_accuracy_score,
          "f1_macro": lambda a, b: f1_score(a, b, average="macro", zero_division=0)}[metric]
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed); n = len(y_true); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        vals.append(fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    return {"point": float(fn(y_true, y_pred)),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "n_boot": len(vals), "n_test": int(n)}

pred_files = find("predictions*.csv")
if pred_files:
    res = []
    for f in pred_files:
        df = pd.read_csv(f)
        tc = next((c for c in df.columns if c.lower() in ("y_true", "true", "label")), None)
        pc = next((c for c in df.columns if c.lower() in ("y_pred", "pred", "prediction")), None)
        if tc is None or pc is None:
            print(f"  [skip] {f}: need y_true and y_pred columns"); continue
        for m in ("accuracy", "balanced_accuracy", "f1_macro"):
            r = bootstrap_ci(df[tc], df[pc], m)
            res.append({"file": os.path.basename(f), "metric": m, **r})
            print(f"  {os.path.basename(f):<34}{m:<20}{r['point']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]")
    if res:
        pd.DataFrame(res).to_csv("bootstrap_cis.csv", index=False)
        print("\nwrote bootstrap_cis.csv")
else:
    print("""[skipped] No per-image prediction files found.

  To enable Part B, the evaluation must save its predictions. In starl_core.evaluate_on_task
  the arrays `y` (true labels) and `p` (predictions) already exist; adding two lines that
  write them to CSV alongside the metrics is sufficient:

      pd.DataFrame({"y_true": y, "y_pred": p}).to_csv(f"predictions_{model}_{task}.csv", index=False)

  Re-running evaluation is inference only and costs minutes. This is worth doing for the
  five single-seed architectures (KAN, GoogLeNet, AlexNet, VGG-11, VGG-19), which currently
  carry no uncertainty estimate at all.
""")
