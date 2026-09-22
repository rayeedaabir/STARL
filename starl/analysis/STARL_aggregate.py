# =============================================================================
# STARL v2 — AGGREGATION NOTEBOOK  (master tables + figures for the paper)
# =============================================================================
# Ingests every task's output zip (the 9 you have), rebuilds the clean master
# results, computes FULL-SEQUENCE CL metrics (ACC / BWT / FM) — which the per-task
# runners could only approximate — and emits paper-ready tables (xlsx + Word) and
# a 300-DPI figure suite + combined PDF.
#
# NO GPU / NO torch needed — pure pandas + matplotlib. Set Accelerator = None.
#
# ATTACH (Add Input) all nine result datasets (upload each zip as a dataset; Kaggle
# extracts them). Filenames inside are unique per task, except the split T4 pair and
# KAN's T4 share names — that's fine, we glob recursively and concat by model.
#   T1_APTOS_outputs, T2_ODIR_outputs, T3_LAG_outputs, T4_HAM_outputs-1/-2,
#   and the four *_kan_outputs.  Everything is matched by the (model, task) inside.
#
# COVERS professor items: #1 (clean 70/15/15 test acc), #7 (baseline ladder),
# #9 (per-class acc/F1 + confusion matrices), #10 (KAN empirical), #14 (Class-IL vs
# Domain-IL). Joint(#7)/CKA(#11)/exemplar(#5)/seeds(#13)/external(#2) come from their
# own notebooks and drop into the same tables later.


# ===== CELL 1 — imports, config, output dirs =====
import os, glob, json, zipfile, shutil, subprocess, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
for pkg, mod in (("openpyxl", "openpyxl"), ("python-docx", "docx")):
    try: __import__(mod)
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

INPUT_ROOT = "/kaggle/input"                 # where the attached datasets live
OUT = "/kaggle/working/aggregate"
for sub in ("csv", "xlsx", "figures", "docx", "tables_docx"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

# Work whether each result zip was attached as its OWN (auto-extracted) dataset OR bundled
# un-extracted inside one dataset: extract any nested *outputs*.zip and search there too.
SEARCH_ROOTS = [INPUT_ROOT]
_EXTRACTED = "/kaggle/working/_extracted"
_nested = glob.glob(f"{INPUT_ROOT}/**/*outputs*.zip", recursive=True)
if _nested:
    for _z in _nested:
        try:
            with zipfile.ZipFile(_z) as zf:
                zf.extractall(os.path.join(_EXTRACTED, os.path.basename(_z)[:-4]))
        except Exception as e:
            print(f"  [warn] could not extract {_z}: {e}")
    SEARCH_ROOTS.append(_EXTRACTED)
    print(f"  extracted {len(_nested)} nested result zip(s) -> searching {_EXTRACTED} too")
DPI = 300
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": DPI, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


# ===== CELL 2 — constants: task order, model order, families, params =====
TASK_ORDER = ["T1_APTOS", "T2_ODIR", "T3_LAG", "T4_HAM"]
TASK_INDEX = {t: i for i, t in enumerate(TASK_ORDER)}
TASK_SHORT = {"T1_APTOS": "T1·APTOS", "T2_ODIR": "T2·ODIR", "T3_LAG": "T3·LAG", "T4_HAM": "T4·HAM"}
TASK_TYPE  = {"T1_APTOS": "base (5 DR)", "T2_ODIR": "Class-IL (+7)",
              "T3_LAG": "Domain-IL (+0)", "T4_HAM": "Class-IL + cross-domain (+7)"}

MODEL_ORDER = ["resnet18", "resnet50", "resnet101", "resnet152", "vgg11", "vgg16", "vgg19",
               "alexnet", "googlenet", "swin_tiny", "coatnet_0", "kan"]
MODEL_FAMILY = {"resnet18": "ResNet", "resnet50": "ResNet", "resnet101": "ResNet", "resnet152": "ResNet",
                "vgg11": "VGG", "vgg16": "VGG", "vgg19": "VGG", "alexnet": "CNN-classic",
                "googlenet": "CNN-classic", "swin_tiny": "Transformer", "coatnet_0": "Transformer",
                "kan": "KAN"}
# total params (millions) — ImageNet backbone reference; overwritten with EXACT counts in
# CELL 9 if starl-code is attached. Used only for the log-scale efficiency scatter.
TOTAL_PARAMS_M = {"resnet18": 11.7, "resnet50": 25.6, "resnet101": 44.5, "resnet152": 60.2,
                  "vgg11": 132.9, "vgg16": 138.4, "vgg19": 143.7, "alexnet": 61.1,
                  "googlenet": 6.6, "swin_tiny": 28.3, "coatnet_0": 27.4, "kan": 12.0}
CL_STRATEGIES = ["naive", "rehearsal", "lwf", "ewc"]        # strategies that track prior tasks
REF_STRATEGIES = ["frozen_probe", "independent"]           # current-task-only references
PROPAGATED = "rehearsal"    # only this strategy's expert is carried to the next task (a true chain)


# ===== CELL 3 — load & normalize every metrics file across all attached zips =====
def _find(pattern):
    hits = []
    for root in SEARCH_ROOTS:
        hits += glob.glob(f"{root}/**/{pattern}", recursive=True)
    return sorted(set(hits))

def _read_all(pattern):
    frames = []
    for f in _find(pattern):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [warn] could not read {f}: {e}"); continue
        # trained_task = the task whose RUN produced this file (from the filename)
        base = os.path.basename(f)
        trained = base.replace(pattern.split("*")[0], "").replace(".csv", "")
        df["trained_task"] = trained
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _coalesce_eval_task(df):
    """T1 runner wrote 'task'; T2-T4/KAN wrote 'eval_task'. After concat BOTH columns can
    exist, so we must COALESCE (not just rename-if-absent) or the T1 rows keep eval_task=NaN
    and the R[0][0] diagonal — the biggest forgetting term — silently vanishes."""
    if "task" in df.columns:
        if "eval_task" in df.columns:
            df["eval_task"] = df["eval_task"].where(df["eval_task"].notna(), df["task"])
        else:
            df = df.rename(columns={"task": "eval_task"})
        df = df.drop(columns=["task"])
    return df

def _norm_final(df):
    if df.empty: return df
    df = _coalesce_eval_task(df)
    return df.drop_duplicates(subset=["trained_task", "model", "experiment", "eval_task"], keep="first")

def _norm_perclass(df):
    if df.empty: return df
    df = _coalesce_eval_task(df)
    if "experiment" not in df.columns:                              # T1 perclass had none -> base
        df["experiment"] = "base"
    return df.drop_duplicates(subset=["trained_task", "model", "experiment", "eval_task", "class"], keep="first")

final_df    = _norm_final(_read_all("final_metrics_*.csv"))
perclass_df = _norm_perclass(_read_all("perclass_*.csv"))
clstep_df   = _read_all("clmetrics_*.csv")        # per-STEP CL metrics the runners wrote

assert not final_df.empty, "No final_metrics_*.csv found — attach the result datasets."
models_seen = [m for m in MODEL_ORDER if m in set(final_df.model)]
tasks_seen  = sorted(set(final_df.trained_task), key=lambda t: TASK_INDEX.get(t, 9))
print("models found :", models_seen, f"({len(models_seen)}/12)")
print("tasks found  :", tasks_seen)
print("experiments  :", sorted(set(final_df.experiment)))
print("final rows   :", len(final_df), "| perclass rows:", len(perclass_df), "| clstep rows:", len(clstep_df))
missing = [m for m in MODEL_ORDER if m not in models_seen]
if missing: print("  [note] missing models (check attachments):", missing)


# ===== CELL 4 — full-sequence CL metrics from the assembled accuracy matrix R =====
def cl_metrics(R):
    """R[i][j] = test acc on task i after training through task j (None if N/A)."""
    N = len(R)
    final = [R[i][N - 1] for i in range(N) if R[i][N - 1] is not None]
    acc = float(np.mean(final)) if final else float("nan")
    bwt, fm = [], []
    for i in range(N - 1):
        if R[i][N - 1] is not None and R[i][i] is not None:
            bwt.append(R[i][N - 1] - R[i][i])
        seen = [R[i][j] for j in range(i, N) if R[i][j] is not None]
        if seen and R[i][N - 1] is not None:
            fm.append(max(seen) - R[i][N - 1])
    return {"average_accuracy": acc,
            "backward_transfer": float(np.mean(bwt)) if bwt else float("nan"),
            "forgetting_measure": float(np.mean(fm)) if fm else float("nan")}

def assemble_R(model, strategy, metric="accuracy"):
    """4x4 R over TASK_ORDER. Column T1 comes from the shared 'base' model; columns T2-T4
    come from `strategy` (rehearsal = a true continual chain; naive/lwf/ewc = per-step
    branches from the maintained model — see the caveat printed in CELL 5)."""
    R = [[None] * 4 for _ in range(4)]
    for j, tj in enumerate(TASK_ORDER):
        exp = "base" if j == 0 else strategy
        sub = final_df[(final_df.model == model) & (final_df.experiment == exp) &
                       (final_df.trained_task == tj)]
        for _, row in sub.iterrows():
            i = TASK_INDEX.get(row.eval_task)
            if i is not None and i <= j and pd.notna(row[metric]):
                R[i][j] = float(row[metric])
    return R

cl_rows, R_store = [], {}
for m in models_seen:
    for s in CL_STRATEGIES:
        R = assemble_R(m, s); R_store[(m, s)] = R
        met = cl_metrics(R)
        cl_rows.append({"model": m, "family": MODEL_FAMILY.get(m, "?"), "strategy": s,
                        "ACC": met["average_accuracy"], "BWT": met["backward_transfer"],
                        "FM": met["forgetting_measure"]})
cl_full = pd.DataFrame(cl_rows)
print("\nFull-sequence CL metrics (head):")
print(cl_full[cl_full.strategy == PROPAGATED].round(4).to_string(index=False))


# ===== CELL 5 — build the paper tables (saved to CSV, one master XLSX, and Word) =====
CAVEAT = ("NOTE: only rehearsal's checkpoint is carried across tasks, so rehearsal is a true "
          "continual chain; naive/LwF/EWC columns are single-step branches from the maintained "
          "model (they isolate each rule's per-step effect). Report rehearsal as the CL result.")
print("\n" + CAVEAT + "\n")

def pivot_final(strategy):
    """models x tasks : test accuracy on task j right after training through task j (diagonal),
    i.e. current-task accuracy per stage."""
    rows = []
    for m in models_seen:
        d = {"model": m}
        for j, tj in enumerate(TASK_ORDER):
            R = R_store.get((m, strategy)) if strategy in CL_STRATEGIES else None
            if R is not None:
                d[TASK_SHORT[tj]] = R[j][j]
            else:
                exp = "base" if j == 0 else strategy
                sub = final_df[(final_df.model == m) & (final_df.experiment == exp) &
                               (final_df.trained_task == tj) & (final_df.eval_task == tj)]
                d[TASK_SHORT[tj]] = float(sub.accuracy.iloc[0]) if len(sub) else np.nan
        rows.append(d)
    return pd.DataFrame(rows)

# Table 1: retained accuracy after the FULL sequence (rehearsal) — the headline forgetting table
t1_rows = []
for m in models_seen:
    R = R_store[(m, PROPAGATED)]
    d = {"model": m, "family": MODEL_FAMILY.get(m, "?")}
    for i, ti in enumerate(TASK_ORDER):
        d[f"acc_{TASK_SHORT[ti]}_final"] = R[i][3]       # accuracy on task i AFTER T4
    met = cl_metrics(R)
    d.update({"ACC": met["average_accuracy"], "BWT": met["backward_transfer"], "FM": met["forgetting_measure"]})
    t1_rows.append(d)
tbl_retention = pd.DataFrame(t1_rows).sort_values("ACC", ascending=False)

# Table 2: current-task accuracy per stage, per strategy (accuracy of each new task when learned)
tbl_currentacc = {s: pivot_final(s) for s in CL_STRATEGIES + REF_STRATEGIES}

# Table 3: full CL-metric comparison across strategies (ACC/BWT/FM)
tbl_clmetrics = cl_full.pivot_table(index=["model", "family"], columns="strategy",
                                    values=["ACC", "BWT", "FM"]).reset_index()

# Table 4: Class-IL (T2) vs Domain-IL (T3) — one-step drop on the immediately-prior task (rehearsal)
di_rows = []
for m in models_seen:
    R = R_store[(m, PROPAGATED)]
    di_rows.append({"model": m,
                    "T2 class-IL: T1 kept": R[0][1], "T3 domain-IL: T2 kept": R[1][2],
                    "T4: T3 kept": R[2][3]})
tbl_scenario = pd.DataFrame(di_rows)

# Table 5: per-class F1 on the FINAL task (T4), rehearsal model — professor #9
pc = perclass_df[(perclass_df.trained_task == "T4_HAM") & (perclass_df.experiment == PROPAGATED) &
                 (perclass_df.eval_task == "T4_HAM")] if not perclass_df.empty else pd.DataFrame()
tbl_perclass_T4 = (pc.pivot_table(index="class", columns="model", values="f1").reset_index()
                   if not pc.empty else pd.DataFrame({"note": ["no T4 perclass rows found"]}))

# Table 6: per-step CL metrics (what the runners logged) averaged across T2-T4 — all strategies
tbl_perstep = (clstep_df.groupby(["model", "experiment"])[["average_accuracy", "backward_transfer",
               "forgetting_measure"]].mean().reset_index() if not clstep_df.empty else pd.DataFrame())

ALL_TABLES = {"retention_after_T4": tbl_retention, "cl_metrics_by_strategy": tbl_clmetrics,
              "scenario_classIL_vs_domainIL": tbl_scenario, "perclass_F1_T4": tbl_perclass_T4,
              "perstep_clmetrics": tbl_perstep}
for s in CL_STRATEGIES + REF_STRATEGIES:
    ALL_TABLES[f"currentacc_{s}"] = tbl_currentacc[s]

def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e:
        import traceback; print(f"  [warn] {step}: {e}"); traceback.print_exc()

# CSVs
for name, df in ALL_TABLES.items():
    _safe(f"csv/{name}.csv", lambda df=df, name=name: df.to_csv(f"{OUT}/csv/{name}.csv", index=False))
# one master XLSX (a sheet per table)
def _write_xlsx():
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    wb = Workbook(); wb.remove(wb.active)
    hdr = PatternFill("solid", start_color="1F3864"); hf = Font(bold=True, color="FFFFFF")
    for name, df in ALL_TABLES.items():
        ws = wb.create_sheet(name[:31])
        cols = [str(c) for c in df.columns]; ws.append(cols)
        for c in range(1, len(cols) + 1):
            ws.cell(1, c).fill = hdr; ws.cell(1, c).font = hf; ws.cell(1, c).alignment = Alignment(horizontal="center")
        for _, r in df.iterrows():
            ws.append([round(v, 4) if isinstance(v, float) else v for v in r.tolist()])
        for i, c in enumerate(cols, 1):
            ws.column_dimensions[get_column_letter(i)].width = max(12, min(34, len(c) + 3))
        ws.freeze_panes = "A2"
    wb.save(f"{OUT}/xlsx/STARL_master_results.xlsx")
_safe("xlsx/STARL_master_results.xlsx", _write_xlsx)


# ===== CELL 6 — figures: forgetting matrices + curves =====
FIG_PATHS = []
def _save(fig, name):
    p = f"{OUT}/figures/{name}.png"; fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    FIG_PATHS.append(p); print(f"  fig {name}")

def fig_forgetting_matrices():
    """R heatmap per model (rehearsal): row = task, col = training stage. Lower-left = retention."""
    ncol = 4; nrow = int(np.ceil(len(models_seen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax in axes[len(models_seen):]: ax.axis("off")
    for k, m in enumerate(models_seen):
        R = np.array([[np.nan if v is None else v for v in row] for row in R_store[(m, PROPAGATED)]])
        ax = axes[k]; im = ax.imshow(R, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels([TASK_SHORT[t] for t in TASK_ORDER], rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(4)); ax.set_yticklabels([TASK_SHORT[t] for t in TASK_ORDER], fontsize=6)
        for i in range(4):
            for j in range(4):
                if not np.isnan(R[i, j]):
                    ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if R[i, j] < 0.6 else "black")
    fig.suptitle("Accuracy matrix R (rehearsal): test acc on task i after training through task j", fontsize=11)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.6, label="test accuracy")
    _save(fig, "01_forgetting_matrices_rehearsal")

def fig_retention_curves():
    """For each task i, its accuracy as training advances T1->T4, naive vs rehearsal (mean over models)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, strat in zip(axes, ["naive", "rehearsal"]):
        for i, ti in enumerate(TASK_ORDER[:-1]):        # tasks that can be forgotten
            ys = []
            for j in range(len(TASK_ORDER)):
                vals = [R_store[(m, strat)][i][j] for m in models_seen if R_store[(m, strat)][i][j] is not None]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(range(4), ys, marker="o", label=f"acc on {TASK_SHORT[ti]}")
        ax.set_title(f"{strat} — prior-task accuracy across the sequence")
        ax.set_xticks(range(4)); ax.set_xticklabels([TASK_SHORT[t] for t in TASK_ORDER])
        ax.set_xlabel("after training through…"); ax.set_ylim(0, 1)
    axes[0].set_ylabel("test accuracy (mean over models)"); axes[1].legend(fontsize=8)
    fig.suptitle("Forgetting curves: naive collapses prior tasks, rehearsal retains", fontsize=11)
    _save(fig, "02_forgetting_curves_naive_vs_rehearsal")

# ===== CELL 7 — figures: strategy & architecture comparisons =====
def fig_clmetrics_bars():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, met, ttl in zip(axes, ["ACC", "BWT", "FM"],
                            ["Average accuracy ↑", "Backward transfer ↑ (0=no forgetting)", "Forgetting measure ↓"]):
        x = np.arange(len(models_seen)); w = 0.2
        for k, s in enumerate(CL_STRATEGIES):
            vals = [cl_full[(cl_full.model == m) & (cl_full.strategy == s)][met].values[0]
                    if len(cl_full[(cl_full.model == m) & (cl_full.strategy == s)]) else np.nan for m in models_seen]
            ax.bar(x + (k - 1.5) * w, vals, w, label=s)
        ax.set_title(ttl, fontsize=10); ax.set_xticks(x)
        ax.set_xticklabels(models_seen, rotation=60, ha="right", fontsize=7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Full-sequence CL metrics across 12 architectures", fontsize=12)
    _save(fig, "03_clmetrics_by_model")

def fig_retention_ranking():
    d = tbl_retention.sort_values("ACC")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"ResNet": "#1f77b4", "VGG": "#ff7f0e", "CNN-classic": "#2ca02c",
              "Transformer": "#d62728", "KAN": "#9467bd"}
    ax.barh(d.model, d.ACC, color=[colors.get(f, "#888") for f in d.family])
    for y, (acc, fm) in enumerate(zip(d.ACC, d.FM)):
        ax.text(acc + 0.005, y, f"{acc:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Average accuracy after the full sequence (rehearsal)")
    ax.set_title("Honest architecture ranking under continual learning")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=f) for f, c in colors.items()], fontsize=8, loc="lower right")
    _save(fig, "04_retention_ranking")

def fig_naive_vs_rehearsal_retention():
    fig, ax = plt.subplots(figsize=(9, 4.6)); x = np.arange(len(models_seen)); w = 0.38
    def prior_avg(strat):
        out = []
        for m in models_seen:
            R = R_store[(m, strat)]; vals = [R[i][3] for i in range(3) if R[i][3] is not None]
            out.append(np.mean(vals) if vals else np.nan)
        return out
    ax.bar(x - w / 2, prior_avg("naive"), w, label="naive", color="#c44")
    ax.bar(x + w / 2, prior_avg("rehearsal"), w, label="rehearsal", color="#4a4")
    ax.set_xticks(x); ax.set_xticklabels(models_seen, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean accuracy on T1–T3 after T4"); ax.set_ylim(0, 1)
    ax.set_title("Prior-task retention after the full sequence: naive vs rehearsal"); ax.legend()
    _save(fig, "05_naive_vs_rehearsal_retention")

def fig_efficiency_frontier():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for m in models_seen:
        acc = tbl_retention[tbl_retention.model == m].ACC.values
        if not len(acc): continue
        p = TOTAL_PARAMS_M.get(m, np.nan)
        ax.scatter(p, acc[0], s=70)
        ax.annotate(m, (p, acc[0]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("total parameters (millions, log)")
    ax.set_ylabel("avg accuracy after sequence (rehearsal)")
    ax.set_title("Memory-efficiency frontier: retention vs model size")
    _save(fig, "06_efficiency_frontier")

def fig_strategy_heatmap():
    piv = cl_full.pivot_table(index="model", columns="strategy", values="ACC").reindex(models_seen)[CL_STRATEGIES]
    fig, ax = plt.subplots(figsize=(6, 6)); im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(CL_STRATEGIES))); ax.set_xticklabels(CL_STRATEGIES, rotation=30, ha="right")
    ax.set_yticks(range(len(models_seen))); ax.set_yticklabels(models_seen, fontsize=8)
    for i in range(len(models_seen)):
        for j in range(len(CL_STRATEGIES)):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                        color="white" if v < 0.5 else "black")
    ax.set_title("Avg accuracy (ACC) by model × strategy"); fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, "07_strategy_model_heatmap")

def fig_family_trends():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    depth = {"resnet18": 18, "resnet50": 50, "resnet101": 101, "resnet152": 152,
             "vgg11": 11, "vgg16": 16, "vgg19": 19}
    for fam, marker in (("ResNet", "o"), ("VGG", "s")):
        ms = [m for m in models_seen if MODEL_FAMILY.get(m) == fam]
        xs = [depth[m] for m in ms]; ys = [tbl_retention[tbl_retention.model == m].ACC.values[0] for m in ms]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], marker=marker, label=fam)
    ax.set_xlabel("network depth (layers)"); ax.set_ylabel("avg accuracy after sequence")
    ax.set_title("Does depth help retention? (ResNet & VGG families)"); ax.legend()
    _save(fig, "08_family_depth_trend")

def fig_kan_spotlight():
    fams = ["ResNet", "VGG", "CNN-classic", "Transformer", "KAN"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, met in zip(axes, ["ACC", "BWT", "FM"]):
        vals = [cl_full[(cl_full.family == f) & (cl_full.strategy == PROPAGATED)][met].mean() for f in fams]
        ax.bar(fams, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        ax.set_title(f"{met} by family (rehearsal)"); ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    fig.suptitle("KAN vs CNN/Transformer families — does spline locality resist forgetting?", fontsize=11)
    _save(fig, "09_kan_spotlight")

def fig_perclass_T4():
    if tbl_perclass_T4.empty or "class" not in tbl_perclass_T4.columns:
        print("  [skip] perclass T4 figure — no data"); return
    piv = tbl_perclass_T4.set_index("class")[[m for m in models_seen if m in tbl_perclass_T4.columns]]
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * piv.shape[1], 1 + 0.4 * piv.shape[0]))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    ax.set_title("Per-class F1 on T4 (rehearsal)"); fig.colorbar(im, ax=ax, shrink=0.7)
    _save(fig, "10_perclass_F1_T4")

for fn in (fig_forgetting_matrices, fig_retention_curves, fig_clmetrics_bars, fig_retention_ranking,
           fig_naive_vs_rehearsal_retention, fig_efficiency_frontier, fig_strategy_heatmap,
           fig_family_trends, fig_kan_spotlight, fig_perclass_T4):
    try: fn()
    except Exception as e:
        import traceback; print(f"  [warn] {fn.__name__}: {e}"); traceback.print_exc()


# ===== CELL 8 — collect the runners' confusion matrices + combined PDF + Word report =====
# copy a curated set of the confusion-matrix PNGs the runners already saved (final task)
cm_out = f"{OUT}/figures/confusion_matrices"; os.makedirs(cm_out, exist_ok=True)
copied = 0
for f in _find("cm_T4_HAM*.png"):
    try: shutil.copy2(f, os.path.join(cm_out, os.path.basename(f))); copied += 1
    except Exception: pass
print(f"  collected {copied} T4 confusion-matrix PNGs")

def _combined_pdf():
    with PdfPages(f"{OUT}/figures/STARL_all_figures.pdf") as pdf:
        for p in FIG_PATHS:
            img = plt.imread(p); fig = plt.figure(figsize=(11, 8.5))
            plt.imshow(img); plt.axis("off"); pdf.savefig(fig, dpi=150); plt.close(fig)
_safe("figures/STARL_all_figures.pdf", _combined_pdf)

def _docx_report():
    import docx
    from docx.shared import Inches
    d = docx.Document()
    d.add_heading("STARL v2 — Aggregated Results", level=0)
    d.add_paragraph(f"Models: {len(models_seen)}/12 · Tasks: {len(tasks_seen)} · "
                    f"Strategies: {', '.join(sorted(set(final_df.experiment)))}")
    d.add_paragraph(CAVEAT)
    def add_table(title, df, maxrows=40):
        d.add_heading(title, level=2)
        df = df.head(maxrows).copy()
        t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Light Grid Accent 1"
        for i, c in enumerate(df.columns): t.rows[0].cells[i].text = str(c)
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for i, c in enumerate(df.columns):
                v = r[c]; cells[i].text = f"{v:.4f}" if isinstance(v, float) else str(v)
    add_table("Table 1 — Retention after the full sequence (rehearsal), ranked", tbl_retention.round(4))
    add_table("Table 2 — Full-sequence CL metrics by strategy", cl_full.round(4))
    add_table("Table 3 — Class-IL vs Domain-IL retention", tbl_scenario.round(4))
    if not tbl_perstep.empty: add_table("Table 4 — Per-step CL metrics (runner logs, mean over T2–T4)", tbl_perstep.round(4))
    d.add_heading("Figures", level=1)
    for p in FIG_PATHS:
        d.add_heading(os.path.basename(p), level=3)
        try: d.add_picture(p, width=Inches(6.2))
        except Exception: pass
    d.save(f"{OUT}/docx/STARL_aggregated_report.docx")
_safe("docx/STARL_aggregated_report.docx", _docx_report)

# individual Word tables (you asked for Word for now)
def _word_tables():
    import docx
    for name, df in ALL_TABLES.items():
        doc = docx.Document(); doc.add_heading(name, level=1)
        df2 = df.round(4) if hasattr(df, "round") else df
        t = doc.add_table(rows=1, cols=len(df2.columns)); t.style = "Light Grid Accent 1"
        for i, c in enumerate(df2.columns): t.rows[0].cells[i].text = str(c)
        for _, r in df2.iterrows():
            cells = t.add_row().cells
            for i, c in enumerate(df2.columns):
                v = r[c]; cells[i].text = f"{v:.4f}" if isinstance(v, float) else str(v)
        doc.save(f"{OUT}/tables_docx/{name}.docx")
_safe("tables_docx/*.docx", _word_tables)


# ===== CELL 9 (optional) — EXACT parameter counts if starl-code is attached =====
# Skips silently if starl-code / torch / timm / pykan aren't available. Overrides the
# approximate TOTAL_PARAMS_M used by the efficiency figure and writes a params table.
def _exact_params():
    import importlib
    cand = _find("starl_core.py")
    if not cand: print("  [skip] starl-code not attached — using reference param counts"); return
    sys.path.append(os.path.dirname(cand[0]))
    import torch  # noqa
    sc = importlib.import_module("starl_core")
    rows = []
    for m in models_seen:
        try:
            model = sc.build_model(m, 19, torch.device("cpu"), 42, pretrained=False, feature_extract=True)
            tot, tr = sc.count_params(model)
            TOTAL_PARAMS_M[m] = tot / 1e6
            rows.append({"model": m, "total_params_M": round(tot / 1e6, 3), "trainable_params_M": round(tr / 1e6, 3)})
            del model
        except Exception as e:
            print(f"  [warn] params {m}: {e}")
    if rows:
        pd.DataFrame(rows).to_csv(f"{OUT}/csv/model_params.csv", index=False)
        print("  wrote csv/model_params.csv (exact)")
_safe("exact params", _exact_params)


# ===== CELL 10 — bundle everything into one downloadable zip =====
manifest = {"models": models_seen, "tasks": tasks_seen,
            "experiments": sorted(set(final_df.experiment)),
            "figures": [os.path.basename(p) for p in FIG_PATHS],
            "tables": list(ALL_TABLES.keys()), "propagated_strategy": PROPAGATED,
            "caveat": CAVEAT}
json.dump(manifest, open(f"{OUT}/aggregate_manifest.json", "w"), indent=2)
out_zip = "/kaggle/working/STARL_aggregate_outputs.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _d, files in os.walk(OUT):
        for fn in files:
            full = os.path.join(root, fn); z.write(full, os.path.relpath(full, OUT))
print("\nBUNDLED ->", out_zip)
print(f"figures: {len(FIG_PATHS)} | tables: {len(ALL_TABLES)} | confusion PNGs: {copied}")
print("\n=== HEADLINE: retention after full sequence (rehearsal), ranked ===")
print(tbl_retention[["model", "family", "ACC", "BWT", "FM"]].round(4).to_string(index=False))
