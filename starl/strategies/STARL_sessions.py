# =============================================================================
# STARL — SESSION-WISE INCREMENTAL ABLATION  (per-session learning curves)
# =============================================================================
# WHY: the earlier exemplar ablation presented the sequence as four large tasks and
# reported one number per configuration. A reviewer sees the destination but not the
# journey. This runs the same images as MANY SMALL SESSIONS, adding roughly one class
# at a time, and records a full metric suite AFTER EVERY SESSION — the reporting
# convention used in few-shot class-incremental learning.
#
# SESSION STRUCTURE (16 sessions, same frozen splits as everything else)
#   0        base session: the 5 diabetic-retinopathy grades          -> 5 classes
#   1..7     one ODIR eye-disease class each                          -> 6..12
#   8        LAG: no new classes, the domain-shift session            -> 12
#   9..15    one HAM skin class each                                  -> 13..19
#
# METRICS PER SESSION (all requested by the reviewer)
#   accuracy, balanced accuracy, macro-F1
#   G-Mean       geometric mean of per-class recall — one ignored class drives it to 0
#   HR@K         hit rate at K: is the true class among the top-K predictions
#   MRR@K        mean reciprocal rank of the true class within the top K
#   RMSE / MAE   ONLY over the ordinal DR grades, where distance is meaningful
#   R matrix     accuracy on every earlier session after every session (16x16)
#   RPD          relative performance drop: (best - current) / best, per session
#
# ATTACH: starl-code, starl-splits, all four image datasets. No expert zip needed —
# the base session is trained here. GPU required.
#
# COST: sessions are individually cheap (one class of data plus replay), but there are
# many. Budget ~4-8 GPU-h per architecture across all exemplar budgets. Start with one
# model and one budget to measure the real per-session time before committing.


# ===== CELL 1 — imports =====
import sys, os, glob, json, time, copy, math
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, torch.nn as nn
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# ===== CELL 2 — config =====
SEED        = 42
EPOCHS      = 10          # per session; sessions are small, 10 is usually enough
LR          = 1e-4
BATCH_SIZE  = 32
IMAGE_SIZE  = 224
WORKERS     = 4
USE_AMP     = True
# FIRST PASS: leave these as-is (resnet18 + BUDGETS=[20]) to measure the real
# per-session time, THEN open BUDGETS back up to [1, 5, 20].
# resnet18 = top retainer, tightest seed variance (sd 0.008), cheapest, and it was in
#            the 4-task exemplar ablation -> the two runs are directly comparable.
# coatnet_0 = the second architecture if GPU allows: a transformer/hybrid shows the
#            curve is not a CNN artefact, and it was also in the 4-task ablation.
# Do NOT run all twelve — the professor's ask is about the reporting PROTOCOL; the
# architecture ranking is already settled by the 12-model, 3-seed main table.
MODELS_TO_RUN = ["resnet18"]          # then ["coatnet_0"] as a second run
BUDGETS       = [20]                  # first pass; widen to [1, 5, 20] once timed
SELECTIONS    = ["herding"]           # random was decisively worse at m=1 and is already
                                      # characterised in the 4-task ablation; omitting it
                                      # halves the grid. Add "random" only if asked.
REH_BATCH     = 4
TOPK          = [1, 3, 5]             # K values for HR@K and MRR@K
SESSION_MAX_MINUTES = 660

WORK = "/kaggle/working/SESSIONS"
for sub in ("csv", "xlsx", "figures", "models"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs = sc.load_class_space(SPLITS_DIR)
ALL, TASKS = cs.all_classes, list(cs.task_order)
NUM_ALL = len(ALL)
print("device:", DEVICE, "| unified classes:", NUM_ALL)


# ===== CELL 3 — build the session plan =====
# Base = every class of T1. Then one class per session from T2, then the T3 domain
# session (no new classes), then one class per session from T4.
def build_sessions():
    plan, seen = [], []
    t1 = sorted(cs.task_idx(TASKS[0]))
    seen += t1
    plan.append({"session": 0, "task": TASKS[0], "new_classes": t1,
                 "seen": list(seen), "kind": "base"})
    s = 1
    for c in sorted(cs.task_idx(TASKS[1])):
        if c in seen: continue
        seen.append(c)
        plan.append({"session": s, "task": TASKS[1], "new_classes": [c],
                     "seen": list(seen), "kind": "class-incremental"}); s += 1
    plan.append({"session": s, "task": TASKS[2], "new_classes": [],
                 "seen": list(seen), "kind": "domain-incremental"}); s += 1
    for c in sorted(cs.task_idx(TASKS[3])):
        if c in seen: continue
        seen.append(c)
        plan.append({"session": s, "task": TASKS[3], "new_classes": [c],
                     "seen": list(seen), "kind": "class-incremental"}); s += 1
    return plan

SESSIONS = build_sessions()
N_SESS = len(SESSIONS)
DR_IDX = sorted(cs.task_idx(TASKS[0]))          # ordinal grades: RMSE/MAE apply here only
print(f"\n{N_SESS} sessions:")
for p in SESSIONS:
    nc = ",".join(ALL[c] for c in p["new_classes"]) or "(none — domain shift)"
    print(f"  S{p['session']:<3}{p['task']:<10}{len(p['seen']):>3} classes seen | +{nc}")
assert DR_IDX == list(range(len(DR_IDX))), "T1 must own unified indices 0..4 for RMSE/MAE"


# ===== CELL 4 — data, filtered per session =====
_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff"); _IDX = None
def REMAP(p):
    global _IDX
    if os.path.exists(p): return p
    if _IDX is None:
        _IDX = {}
        for r, _d, fs in os.walk("/kaggle/input"):
            for f in fs:
                if f.lower().endswith(_EXT): _IDX.setdefault(f, os.path.join(r, f))
        print(f"  indexed {len(_IDX)} images")
    return _IDX.get(os.path.basename(p), p)

train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def rows_of(task, split, classes=None):
    r = sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split)
    return [x for x in r if classes is None or x[1] in classes]
def mk(rows, tf, batch, shuffle):
    return sc._loader(rows, tf, batch, shuffle, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
for t in TASKS:
    assert os.path.exists(REMAP(rows_of(t, "train")[0][0])), f"{t}: images missing"

# Evaluation pools for a session = every class seen so far, gathered across the tasks that
# own them. VAL drives checkpoint selection; TEST is only ever read for reporting.
def _rows_for(seen, split):
    out = []
    for t in TASKS:
        out += rows_of(t, split, set(seen) & set(cs.task_idx(t)))
    return out
def val_rows_for(seen):  return _rows_for(seen, "val")    # selection ONLY
def test_rows_for(seen): return _rows_for(seen, "test")   # reporting ONLY
print("  data helpers ready")


# ===== CELL 5 — the metric suite =====
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)

def g_mean(y, p, labels):
    """Geometric mean of per-class recall. Zero if any class is never recovered —
    which is exactly the behaviour that makes it useful on imbalanced medical data."""
    r = recall_score(y, p, labels=labels, average=None, zero_division=0)
    r = np.asarray(r, dtype=float)
    if len(r) == 0: return float("nan")
    return float(np.exp(np.mean(np.log(np.clip(r, 1e-12, None))))) if (r > 0).all() else 0.0

def topk_metrics(probs, y, allowed, ks):
    """HR@K and MRR@K restricted to the classes seen so far.
    HR@K  = fraction of images whose true class is in the top K.
    MRR@K = mean of 1/rank when the true class is in the top K, else 0."""
    out = {}
    sub = probs[:, allowed]                      # rank only among classes seen so far
    order = np.argsort(-sub, axis=1)             # descending
    pos = {c: i for i, c in enumerate(allowed)}
    tgt = np.array([pos[v] for v in y])
    rank = np.array([np.where(order[i] == tgt[i])[0][0] + 1 for i in range(len(y))])
    for k in ks:
        if k > len(allowed): continue
        out[f"HR@{k}"] = float(np.mean(rank <= k))
        out[f"MRR@{k}"] = float(np.mean(np.where(rank <= k, 1.0 / rank, 0.0)))
    return out

def ordinal_errors(y, p, dr_idx):
    """RMSE and MAE over the DR grades ONLY. These treat the grade as a number, which
    is meaningful for severity 0-4 and meaningless for unordered lesion categories, so
    images of other classes are excluded rather than silently mis-scored."""
    m = np.isin(y, dr_idx) & np.isin(p, dr_idx)
    if m.sum() == 0: return {"RMSE_dr": float("nan"), "MAE_dr": float("nan"), "n_dr": 0}
    d = (y[m] - p[m]).astype(float)
    return {"RMSE_dr": float(np.sqrt(np.mean(d ** 2))), "MAE_dr": float(np.mean(np.abs(d))),
            "n_dr": int(m.sum())}

@torch.no_grad()
def evaluate(model, loader, allowed):
    """Predictions restricted to the classes seen so far, plus the full metric suite."""
    model.eval(); P, Y = [], []
    al = torch.tensor(sorted(allowed), device=DEVICE)
    for images, labels in loader:
        out = model(images.to(DEVICE))
        masked = torch.full_like(out, float("-inf")); masked[:, al] = out[:, al]
        P.append(torch.softmax(out, 1).cpu().numpy()); Y.append(labels.numpy())
    probs = np.concatenate(P); y = np.concatenate(Y)
    allowed_s = sorted(allowed)
    sub = probs[:, allowed_s]
    p = np.array(allowed_s)[sub.argmax(1)]
    m = {"accuracy": accuracy_score(y, p),
         "balanced_accuracy": balanced_accuracy_score(y, p),
         "f1_macro": f1_score(y, p, average="macro", zero_division=0),
         "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
         "G_Mean": g_mean(y, p, allowed_s), "n": int(len(y))}
    m.update(topk_metrics(probs, y, allowed_s, TOPK))
    m.update(ordinal_errors(y, p, DR_IDX))
    return m, y, p


# ===== CELL 6 — exemplar selection (herding / random), reused from the ablation =====
def _feature_extractor(model, name):
    feats = {}
    target = sc._head_module(model, name)
    def hook(_m, inp, _o):
        x = inp[0]
        if x.dim() > 2: x = torch.flatten(x, 1)
        feats["f"] = x.detach().float()
    h = target.register_forward_hook(hook)
    def extract(images):
        with torch.no_grad(): model(images.to(DEVICE))
        return feats["f"]
    return extract, h

def select_exemplars(model, name, task, classes, m, method):
    """Up to m images per class from `task`'s TRAIN split, for the given classes."""
    rows = rows_of(task, "train", set(classes))
    by = {}
    for pth, lab in rows: by.setdefault(lab, []).append((pth, lab))
    rng = np.random.default_rng(SEED); chosen = []
    if method == "random":
        for lab, items in by.items():
            idx = rng.choice(len(items), size=min(m, len(items)), replace=False)
            chosen += [items[i] for i in idx]
        return chosen
    extract, handle = _feature_extractor(model, name)
    try:
        for lab, items in by.items():
            if len(items) <= m: chosen += items; continue
            F = []
            for images, _l in mk(items, eval_tf, BATCH_SIZE, False): F.append(extract(images).cpu())
            F = torch.cat(F).numpy()
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
            mu = F.mean(0); picked, run = [], np.zeros_like(mu)
            for k in range(min(m, len(items))):
                d = np.linalg.norm(mu - (run[None, :] + F) / (k + 1), axis=1)
                d[picked] = np.inf
                j = int(np.argmin(d)); picked.append(j); run = run + F[j]
            chosen += [items[j] for j in picked]
    finally:
        handle.remove()
    return chosen


# ===== CELL 7 — run one configuration through all sessions =====
logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_sessions.csv")
session_rows, R_store, perclass_rows = [], {}, []
START = time.time()

class _Budget(Exception): pass

def run_config(name, m, method):
    tag = f"{name}|m={m}|{method}"
    print("\n" + "=" * 78 + f"\n### {tag}\n" + "=" * 78, flush=True)
    R = [[None] * N_SESS for _ in range(N_SESS)]     # R[i][j] = acc on session i's classes after session j
    buffers, model, best_seen = {}, None, {}
    for p in SESSIONS:
        s, task, seen = p["session"], p["task"], p["seen"]
        t0 = time.time()
        # fixed 19-wide head so head size never confounds the session comparison
        student = sc.build_model(name, NUM_ALL, DEVICE, SEED,
                                 pretrained=(s == 0), feature_extract=True)
        if model is not None:
            student, _ = sc.transfer_backbone(model, student, name)
        # training data = this session's new classes (or, for the domain session, LAG's data)
        cls_now = p["new_classes"] if p["new_classes"] else sorted(cs.task_idx(task))
        tr_rows = rows_of(task, "train", set(cls_now))
        tr = mk(tr_rows, train_tf, BATCH_SIZE, True)
        reh = [mk(v, train_tf, REH_BATCH, True) for v in buffers.values() if v]
        opt = torch.optim.Adam(filter(lambda q: q.requires_grad, student.parameters()), lr=LR)
        # Seen-class VALIDATION pool. train_rehearsal/train_naive fall back to eval_loaders
        # for best-checkpoint selection when select_loaders is None, so this MUST be val —
        # passing the test rows here would select the checkpoint on the frozen test set.
        va = mk(val_rows_for(seen), eval_tf, BATCH_SIZE, False)
        if reh:
            student, _ = sc.train_rehearsal(student, tr, reh, {"seen": va}, opt, DEVICE, EPOCHS,
                                            f"{name}_s{s}", logger=logger, select_loaders=None,
                                            best_ckpt_path=None, all_classes=ALL, seed=SEED,
                                            use_amp=USE_AMP)
        else:
            student, _ = sc.train_naive(student, tr, {"seen": va}, opt, DEVICE, EPOCHS,
                                        f"{name}_s{s}", logger=logger, experiment=f"session{s}",
                                        select_best=False, all_classes=ALL, seed=SEED, use_amp=USE_AMP)
        # ---- evaluate on everything seen so far ----
        te = mk(test_rows_for(seen), eval_tf, BATCH_SIZE, False)
        met, y, pr = evaluate(student, te, seen)
        # ---- fill the R matrix: accuracy on each earlier session's own classes ----
        for i, q in enumerate(SESSIONS[:s + 1]):
            cls_i = q["new_classes"] if q["new_classes"] else sorted(cs.task_idx(q["task"]))
            mask = np.isin(y, cls_i)
            if mask.sum(): R[i][s] = float((y[mask] == pr[mask]).mean())
        # ---- RPD against each session's own best ----
        for i in range(s + 1):
            if R[i][s] is not None:
                best_seen[i] = max(best_seen.get(i, 0.0), R[i][s])
        rpd = [ (best_seen[i] - R[i][s]) / best_seen[i]
                for i in range(s) if R[i][s] is not None and best_seen.get(i, 0) > 0 ]
        row = {"model": name, "m": m, "selection": method, "session": s, "task": task,
               "kind": p["kind"], "classes_seen": len(seen),
               "new_classes": ";".join(ALL[c] for c in p["new_classes"]),
               **met, "RPD": float(np.mean(rpd)) if rpd else 0.0,
               "minutes": round((time.time() - t0) / 60, 2)}
        session_rows.append(row)
        for c in sorted(seen):
            mk_ = (y == c)
            if mk_.sum():
                perclass_rows.append({"model": name, "m": m, "selection": method, "session": s,
                                      "class": ALL[c], "recall": float((pr[mk_] == c).mean()),
                                      "support": int(mk_.sum())})
        print(f"  S{s:<3}{len(seen):>3} cls | acc {met['accuracy']:.4f} bal {met['balanced_accuracy']:.4f} "
              f"G {met['G_Mean']:.4f} F1m {met['f1_macro']:.4f} "
              f"HR@3 {met.get('HR@3', float('nan')):.3f} MRR@3 {met.get('MRR@3', float('nan')):.3f} "
              f"RPD {row['RPD']:.3f} | {row['minutes']:.1f}m", flush=True)
        # ---- update the replay buffer with this session's classes ----
        if cls_now:
            ex = select_exemplars(student, name, task, cls_now, m, method)
            buffers[f"{task}_s{s}"] = ex
        del model; model = student
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if (time.time() - START) / 60 > SESSION_MAX_MINUTES:
            print("  [budget] session limit reached — rerun for the remaining configs.", flush=True)
            R_store[tag] = R; raise _Budget()
    R_store[tag] = R
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

try:
    for name in MODELS_TO_RUN:
        for m in BUDGETS:
            for method in SELECTIONS:
                try:
                    run_config(name, m, method)
                except _Budget: raise
                except Exception as e:
                    import traceback; print(f"  [ERROR] {name}/m={m}/{method}: {e}"); traceback.print_exc()
except _Budget:
    print("\n*** stopped on the session budget — partial results are saved below ***")


# ===== CELL 8 — outputs =====
df = pd.DataFrame(session_rows)
def _safe(s, f):
    try: f(); print("  wrote", s)
    except Exception as e: print(f"  [warn] {s}: {e}")
_safe("sessions_metrics.csv", lambda: df.to_csv(f"{WORK}/csv/sessions_metrics.csv", index=False))
_safe("sessions_perclass.csv", lambda: pd.DataFrame(perclass_rows).to_csv(f"{WORK}/csv/sessions_perclass.csv", index=False))
_safe("sessions_R.json", lambda: json.dump(
    {"sessions": [{k: v for k, v in p.items() if k != "seen"} for p in SESSIONS], "R": R_store},
    open(f"{WORK}/csv/sessions_R.json", "w"), indent=2))
_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"per_session": session_rows, "per_class": perclass_rows}, f"{WORK}/xlsx/SESSIONS_results.xlsx"))

def fig_curves():
    if df.empty: return
    mets = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
            ("G_Mean", "G-Mean"), ("f1_macro", "Macro-F1")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (c, lab) in zip(axes.ravel(), mets):
        for (mm, sel), g in df.groupby(["m", "selection"]):
            g = g.sort_values("session")
            ax.plot(g.session, g[c], marker="o", ms=3.5, label=f"m={mm} {sel}")
        ax.set_title(lab, fontsize=10); ax.set_ylim(0, 1); ax.grid(alpha=.3)
    for ax in axes[1]: ax.set_xlabel("session")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Performance after every incremental session", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/session_curves.png", dpi=300); plt.close(fig)
    print("  fig session_curves")

def fig_topk_rpd():
    if df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (mm, sel), g in df.groupby(["m", "selection"]):
        g = g.sort_values("session")
        for k in TOPK:
            if f"HR@{k}" in g: axes[0].plot(g.session, g[f"HR@{k}"], marker="o", ms=3, label=f"HR@{k} m={mm}")
        axes[1].plot(g.session, g["RPD"], marker="s", ms=3, label=f"m={mm} {sel}")
    axes[0].set_title("Hit rate at K"); axes[1].set_title("Relative performance drop ↓")
    for a in axes: a.set_xlabel("session"); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/session_topk_rpd.png", dpi=300); plt.close(fig)
    print("  fig session_topk_rpd")
_safe("figures", lambda: (fig_curves(), fig_topk_rpd()))

zip_path = sc.bundle_outputs("SESSIONS", WORK, out_zip="/kaggle/working/SESSIONS_outputs.zip",
                             run_manifest={"experiment": "session_wise_incremental",
                                           "n_sessions": N_SESS, "models": MODELS_TO_RUN,
                                           "budgets": BUDGETS, "selections": SELECTIONS,
                                           "epochs_per_session": EPOCHS, "topk": TOPK, "seed": SEED})
print("\nBUNDLED ->", zip_path)
if not df.empty:
    cols = ["session", "classes_seen", "kind", "accuracy", "balanced_accuracy", "G_Mean",
            "f1_macro"] + [f"HR@{k}" for k in TOPK if f"HR@{k}" in df] + ["RPD", "RMSE_dr", "MAE_dr"]
    print("\n=== PER-SESSION RESULTS (this is the table the reviewer asked for) ===")
    for (nm, mm, sel), g in df.groupby(["model", "m", "selection"]):
        print(f"\n--- {nm}, m={mm}, {sel} ---")
        print(g[cols].round(4).to_string(index=False))
