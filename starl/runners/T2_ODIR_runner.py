# =============================================================================
# STARL v2 — TASK 2+ (CL transition) RUNNER   [default: T2_ODIR]
# =============================================================================
# This is the TEMPLATED T2–T4 continual-learning runner. It is written for T2
# (APTOS -> ODIR) out of the box; to run T3 or T4 later you only change TASK in
# CELL 2 and swap the attached "expert" dataset (see the header of CELL 2).
#
# HOW TO USE ON KAGGLE:
#   Each "# ===== CELL n =====" block below = ONE notebook cell. Paste each block
#   into its own cell, top to bottom, then validate one model, then Save & Run All.
#
# ATTACH TO THE NOTEBOOK (right sidebar -> Add Input):
#   1. starl-code        — dataset with starl_core.py + starl_baselines.py (UNCHANGED
#                          from T1; this runner imports it, does not modify it)
#   2. starl-splits      — the Phase-1 output (unzipped)
#   3. T1 outputs        — upload T1_APTOS_outputs.zip as a Kaggle dataset; it holds
#                          models/expert_T1_APTOS_<model>.pth (the experts T2 loads)
#   4. image datasets    — the SAME APTOS + ODIR datasets used in Phase 1
#                          (T3 also needs LAG; T4 also needs HAM10000)
#   You do NOT hand-edit any image path: CELL 3 auto-resolves every manifest path
#   by filename no matter where each dataset is mounted.
#
# WHAT IT DOES (per model), reusing starl_core + starl_baselines unchanged:
#   * load expert_<prev>_<model>.pth  -> evaluate it on every PRIOR task (ceiling
#     BEFORE this task = the reference row of the accuracy matrix)
#   * transfer that backbone into a grown-head model (num_classes = through THIS task)
#   * train + evaluate the strategies: naive, rehearsal, LwF, EWC, frozen-probe,
#     independent  (rehearsal's best-by-val checkpoint becomes the NEXT task's expert)
#   * per (model, strategy) build the accuracy matrix R over [priors..., current] and
#     compute CL metrics (Average Accuracy / Backward Transfer / Forgetting Measure)
#   * bundle models + CSV + XLSX + DOCX + figures + the raw R matrices into one zip


# ===== CELL 1 — imports & attach the code library =====
import sys, os, json, time, glob
CODE_DIR   = "/kaggle/input/starl-code"      # folder with starl_core.py + starl_baselines.py
SPLITS_DIR = "/kaggle/input/starl-splits"    # Phase-1 output (unzipped)
sys.path.append(CODE_DIR)
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
import starl_baselines as sb                 # LwF / EWC live here — used from T2 on
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
# python-docx is not preinstalled on Kaggle — install quietly (harmless if present/offline)
try:
    import docx  # noqa
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"])


# ===== CELL 2 — config =====
# ---- change ONLY these three lines to move to T3 / T4 -----------------------
#   T3_LAG : TASK="T3_LAG",  attach the T2 output zip (expert_T2_ODIR_*.pth) + LAG
#   T4_HAM : TASK="T4_HAM",  attach the T3 output zip (expert_T3_LAG_*.pth)  + HAM
TASK        = "T2_ODIR"
EXPERT_DIR  = None          # None = auto-find the folder holding expert_<prev>_*.pth
# ----------------------------------------------------------------------------
SEED        = 42
EPOCHS      = 15
LR          = 1e-4
BATCH_SIZE  = 32
IMAGE_SIZE  = 224
WORKERS     = 4
USE_AMP     = True          # AMP + cuDNN autotuner (KAN runner overrides this to False)

# Validate the WHOLE pipeline on ONE model first (fast), then switch to the full 11:
MODELS_TO_RUN = ["resnet50"]
# MODELS_TO_RUN = ["resnet18","resnet50","resnet101","resnet152",
#                  "vgg11","vgg16","vgg19","alexnet","googlenet","swin_tiny","coatnet_0"]

# Which strategies to run. naive+rehearsal+lwf+ewc build the CL accuracy matrix;
# frozen-probe + independent are current-task reference points; joint is the global
# upper bound (kept OFF here — run it once in the aggregation notebook over all tasks).
RUN_NAIVE        = True
RUN_REHEARSAL    = True      # its best-by-val checkpoint = the next task's expert
RUN_LWF          = True
RUN_EWC          = True
RUN_FROZEN_PROBE = True       # previous-expert backbone frozen, only the new head trains
RUN_INDEPENDENT  = True       # fresh ImageNet model on THIS task only (per-task ceiling)
RUN_JOINT        = False      # train on all-tasks-so-far together (expensive; usually 1x)

# strategy hyper-params
REHEARSAL_FRACTION = 0.10     # stratified % of each prior task's TRAIN split, replayed
REH_BATCH          = 4        # replay images added per new-task batch, PER prior task
LWF_T, LWF_LAM     = 2.0, 1.0
EWC_LAM            = 1000.0
EWC_FISHER_BATCHES = 200      # cap batches when estimating Fisher (per prior task)
PLOT_CM_STRATEGIES = {"naive", "rehearsal"}   # which strategies get a confusion-matrix PNG

# WORK is the OUTPUT folder. It MUST be under /kaggle/working (writable).
WORK = f"/kaggle/working/{TASK}"
for sub in ("models", "csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
# 224px image cache: built once on the first epoch, reused by every epoch AND every model
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print("device:", DEVICE, "| task:", TASK, "| output ->", WORK)


# ===== CELL 3 — class space, prior tasks, auto path-resolver, current-task data =====
cs       = sc.load_class_space(SPLITS_DIR)
NUM      = cs.num_classes_through(TASK)      # head size THROUGH this task (T2 -> 12)
TASK_IDX = cs.task_idx(TASK)                 # unified indices of THIS task's classes
PRIOR    = cs.prior_tasks(TASK)              # e.g. ["T1_APTOS"]; ordered oldest->newest
assert PRIOR, f"{TASK} has no prior task — use the T1 runner for the base session."
PREV     = PRIOR[-1]                         # immediately previous task (its expert we load)
SEQ      = PRIOR + [TASK]                    # task order used to build the accuracy matrix R
print(f"{TASK}: {NUM} classes | prior tasks: {PRIOR} | prev expert: {PREV}")

# The manifest stores absolute image paths from Phase 1. If a dataset is mounted at a
# different path now, resolve each image by filename against a one-time index of
# everything under /kaggle/input (handles MANY datasets at once — APTOS + ODIR + ...).
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_INPUT_IDX = None
def _build_input_index(root="/kaggle/input"):
    idx = {}
    for r, _d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(_IMG_EXT):
                idx.setdefault(f, os.path.join(r, f))   # first hit wins; basenames are unique
    return idx
def REMAP(path):
    """Manifest path -> real on-disk path. Returns path unchanged if it already resolves."""
    if os.path.exists(path):
        return path
    global _INPUT_IDX
    if _INPUT_IDX is None:
        print("  building /kaggle/input image index (first path miss)...", flush=True)
        _INPUT_IDX = _build_input_index()
        print(f"  indexed {len(_INPUT_IDX)} images", flush=True)
    return _INPUT_IDX.get(os.path.basename(path), path)

# sanity-check that current + every prior task's images resolve
for t in SEQ:
    _rows = sc.read_manifest(sc.manifest_path(SPLITS_DIR, t))
    _p0 = _rows[0][0]
    _res = REMAP(_p0)
    assert os.path.exists(_res), (
        f"{t}: image not found ({os.path.basename(_p0)}). "
        f"Attach the SAME source dataset used in Phase 1 for {t}.")
print("  all task sample images resolve OK")

# current-task loaders (cached + remapped)
tr_loader, va_loader, te_loader = sc.build_task_loaders(
    SPLITS_DIR, TASK, BATCH_SIZE, IMAGE_SIZE, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
print(f"  {TASK}: train {len(tr_loader.dataset)} | val {len(va_loader.dataset)} | "
      f"test {len(te_loader.dataset)} | classes {cs.task_classes[TASK]}")

# resolve where the previous experts live
if EXPERT_DIR is None:
    _hits = glob.glob(f"/kaggle/input/**/expert_{PREV}_*.pth", recursive=True)
    EXPERT_DIR = os.path.dirname(_hits[0]) if _hits else None
assert EXPERT_DIR and os.path.isdir(EXPERT_DIR), (
    f"Could not find expert_{PREV}_*.pth under /kaggle/input. "
    f"Upload {PREV}'s output zip as a dataset and attach it.")
print("  expert dir:", EXPERT_DIR)


# ===== CELL 4 — prior-task loaders (val / test / rehearsal / fisher) =====
train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def _mkloader(task, split, tf, batch, shuffle):
    rows = sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split)
    return sc._loader(rows, tf, batch, shuffle, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)

# eval_loaders passed to every training loop = VAL of ALL seen tasks (uniform CSV columns,
# one eval pass/epoch, gives the per-epoch forgetting curve). Current task first.
ALL_VAL  = {TASK: va_loader}
ALL_TEST = {TASK: te_loader}
prior_val, prior_test, rehearsal, prior_train = {}, {}, {}, {}
for p in PRIOR:
    prior_val[p]  = _mkloader(p, "val",  eval_tf, BATCH_SIZE, False)
    prior_test[p] = _mkloader(p, "test", eval_tf, BATCH_SIZE, False)
    ALL_VAL[p]  = prior_val[p]
    ALL_TEST[p] = prior_test[p]
    rehearsal[p], _n = sc.build_rehearsal_loader(
        SPLITS_DIR, p, fraction=REHEARSAL_FRACTION, reh_batch=REH_BATCH,
        image_size=IMAGE_SIZE, workers=WORKERS, seed=SEED, path_remap=REMAP)
    if RUN_EWC:
        prior_train[p] = _mkloader(p, "train", eval_tf, BATCH_SIZE, True)   # Fisher source
    print(f"  prior {p}: val {len(prior_val[p].dataset)} | test {len(prior_test[p].dataset)} "
          f"| rehearsal {_n}")


# ===== CELL 5 — plotting helpers =====
def plot_confusion(cm, class_names, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.4 + 0.6 * len(class_names), 1.2 + 0.6 * len(class_names)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def plot_retention(name, expert_acc, after_by_strategy, path):
    """Bar chart: prior-task accuracy before (expert) vs after each strategy — shows forgetting."""
    try:
        tasks = PRIOR
        strategies = list(after_by_strategy.keys())
        x = np.arange(len(tasks)); w = 0.8 / (len(strategies) + 1)
        fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(tasks), 3))
        ax.bar(x, [expert_acc[t] for t in tasks], w, label="expert (before)")
        for k, s in enumerate(strategies, 1):
            ax.bar(x + k * w, [after_by_strategy[s].get(t, 0.0) for t in tasks], w, label=s)
        ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(tasks, fontsize=8)
        ax.set_ylabel("accuracy"); ax.set_ylim(0, 1); ax.set_title(f"{name} — prior-task retention", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    except Exception as e:
        print(f"  [warn] retention plot for {name} skipped: {e}")


# ===== CELL 6 — result accumulators + per-model runner =====
epoch_logger  = sc.MetricLogger(f"{WORK}/csv/epoch_history_{TASK}.csv")
final_rows    = []     # one row per (model, experiment, eval_task)
perclass_rows = []     # one row per (model, experiment, eval_task, class)
clmetrics_rows= []     # one row per (model, experiment) for CL strategies
cl_matrices   = {}     # {model: {strategy: R}}  raw matrices for the aggregation notebook

def _adam(model):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

def _load_expert(name):
    """Build the previous model at ITS head size and load its trained weights."""
    ckpt = torch.load(os.path.join(EXPERT_DIR, f"expert_{PREV}_{name}.pth"), map_location=DEVICE)
    expert = sc.build_model(name, cs.num_classes_through(PREV), DEVICE, SEED,
                            pretrained=False, feature_extract=True)
    expert.load_state_dict(ckpt["model_state_dict"])
    expert.eval()
    return expert

def _fresh_student(name, expert, frozen_probe=False):
    """Grown-head model (NUM classes) with the expert's backbone transferred in.
    pretrained=False is safe: the backbone is overwritten by transfer, the head is fresh."""
    student = sc.build_model(name, NUM, DEVICE, SEED, pretrained=False,
                             feature_extract=True, frozen_probe=frozen_probe)
    student, _ = sc.transfer_backbone(expert, student, name)
    return student

def _eval(model, task):
    return sc.evaluate_on_task(model, ALL_TEST[task], cs.task_idx(task), cs.all_classes, DEVICE)

def _record(name, experiment, eval_task, m, minutes=None):
    final_rows.append({"model": name, "experiment": experiment, "eval_task": eval_task,
                       "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                       "f1_macro": m["f1_macro"], "precision_weighted": m["precision_weighted"],
                       "recall_weighted": m["recall_weighted"],
                       "roc_auc": m["roc_auc_ovr_weighted"], "train_minutes": minutes})
    for cls, v in m["per_class"].items():
        perclass_rows.append({"model": name, "experiment": experiment, "eval_task": eval_task,
                              "class": cls, **v})

def run_model(name):
    print("\n" + "=" * 74 + f"\n### MODEL: {name}   (task {TASK})\n" + "=" * 74, flush=True)
    expert = _load_expert(name)

    # (1) expert ceiling on every prior task = the "before this task" reference row of R
    expert_acc = {}
    for p in PRIOR:
        m = _eval(expert, p); expert_acc[p] = m["accuracy"]
        _record(name, "expert_before", p, m)
        print(f"  expert on {p}: acc {m['accuracy']:.4f}", flush=True)

    # (2) EWC Fisher — compute NOW, before LwF flips the expert's requires_grad off
    fisher_list = star_list = None
    if RUN_EWC:
        fisher_list, star_list = [], []
        for p in PRIOR:
            f, s = sb.compute_fisher(expert, prior_train[p], DEVICE, max_batches=EWC_FISHER_BATCHES)
            fisher_list.append(f); star_list.append(s)
        print(f"  fisher estimated on {len(PRIOR)} prior task(s)", flush=True)

    after_by_strategy = {}   # strategy -> {task: acc} (for the retention plot)

    def _finish(strategy, model, minutes):
        """Evaluate a trained CL model on every seen task, build R, log CL metrics."""
        after = {}; m_cur = None
        for t in SEQ:
            m = _eval(model, t); after[t] = m["accuracy"]
            _record(name, strategy, t, m, minutes if t == TASK else None)
            if t == TASK:
                m_cur = m
        after_by_strategy[strategy] = after
        if strategy in PLOT_CM_STRATEGIES and m_cur is not None:
            plot_confusion(m_cur["confusion_matrix"], m_cur["class_names"],
                           f"{name} — {TASK} {strategy} (test)",
                           f"{WORK}/figures/cm_{TASK}_{name}_{strategy}.png")
        # accuracy matrix R over SEQ (priors..., current). Only entries THIS transition
        # can fill are set; the aggregation notebook stitches full-sequence R from these.
        N = len(SEQ); R = [[None] * N for _ in range(N)]
        for i, ti in enumerate(SEQ):
            R[i][N - 1] = after[ti]                 # accuracy after the CURRENT task
            if ti in expert_acc:                    # prior rows: accuracy after PREV (reference)
                R[i][N - 2] = expert_acc[ti]
        clm = sc.cl_metrics(R)
        cl_matrices.setdefault(name, {})[strategy] = R
        clmetrics_rows.append({"model": name, "experiment": strategy,
                               "average_accuracy": clm["average_accuracy"],
                               "backward_transfer": clm["backward_transfer"],
                               "forgetting_measure": clm["forgetting_measure"]})
        print(f"  [{strategy}] ACC {clm['average_accuracy']:.4f} "
              f"BWT {clm['backward_transfer']:.4f} FM {clm['forgetting_measure']:.4f}", flush=True)

    # (3) NAIVE — fine-tune on current task only; select best CURRENT-task val checkpoint
    if RUN_NAIVE:
        t0 = time.time(); model = _fresh_student(name, expert)
        model, _ = sc.train_naive(
            model, tr_loader, eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE,
            epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="naive",
            select_best=True, select_loaders={TASK: va_loader},
            all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        _finish("naive", model, round((time.time() - t0) / 60, 2))
        del model; torch.cuda.empty_cache()

    # (4) REHEARSAL — new task + replay of every prior task; best-by-ALL-val -> next expert
    if RUN_REHEARSAL:
        t0 = time.time(); model = _fresh_student(name, expert)
        ckpt = f"{WORK}/models/expert_{TASK}_{name}.pth"     # <- the file the NEXT task loads
        model, _ = sc.train_rehearsal(
            model, new_task_loader=tr_loader, rehearsal_loaders=[rehearsal[p] for p in PRIOR],
            eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE, epochs=EPOCHS,
            model_name=name, logger=epoch_logger, select_loaders=None, best_ckpt_path=ckpt,
            all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        _finish("rehearsal", model, round((time.time() - t0) / 60, 2))
        del model; torch.cuda.empty_cache()

    # (5) LwF — distil the expert's old-class logits (no stored old data)
    if RUN_LWF:
        t0 = time.time(); model = _fresh_student(name, expert)
        model, _ = sb.train_lwf(
            model, teacher=expert, train_loader=tr_loader, eval_loaders=ALL_VAL,
            optimizer=_adam(model), device=DEVICE, epochs=EPOCHS, model_name=name,
            n_old_classes=cs.num_classes_through(PREV), T=LWF_T, lam=LWF_LAM,
            logger=epoch_logger, select_loaders=None, all_classes=cs.all_classes, seed=SEED)
        _finish("lwf", model, round((time.time() - t0) / 60, 2))
        del model; torch.cuda.empty_cache()

    # (6) EWC — Fisher-weighted penalty keeping shared backbone near its prior values
    if RUN_EWC:
        t0 = time.time(); model = _fresh_student(name, expert)
        model, _ = sb.train_ewc(
            model, tr_loader, eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE,
            epochs=EPOCHS, model_name=name, fisher_list=fisher_list, star_list=star_list,
            lam=EWC_LAM, logger=epoch_logger, select_loaders=None,
            all_classes=cs.all_classes, seed=SEED)
        _finish("ewc", model, round((time.time() - t0) / 60, 2))
        del model; torch.cuda.empty_cache()

    # (7) FROZEN PROBE — expert backbone frozen, only the new head trains (current task ref)
    if RUN_FROZEN_PROBE:
        t0 = time.time(); model = _fresh_student(name, expert, frozen_probe=True)
        model, _ = sc.train_naive(
            model, tr_loader, eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE,
            epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="frozen_probe",
            select_best=True, select_loaders={TASK: va_loader},
            all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        m = _eval(model, TASK); _record(name, "frozen_probe", TASK, m, round((time.time() - t0) / 60, 2))
        print(f"  [frozen_probe] {TASK} acc {m['accuracy']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    # (8) INDEPENDENT — fresh ImageNet model trained on THIS task only (per-task ceiling)
    if RUN_INDEPENDENT:
        t0 = time.time()
        model = sc.build_model(name, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
        model, _ = sc.train_naive(
            model, tr_loader, eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE,
            epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="independent",
            select_best=True, select_loaders={TASK: va_loader},
            all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        m = _eval(model, TASK); _record(name, "independent", TASK, m, round((time.time() - t0) / 60, 2))
        print(f"  [independent] {TASK} acc {m['accuracy']:.4f}", flush=True)
        del model; torch.cuda.empty_cache()

    # retention figure (prior-task acc: expert vs each CL strategy)
    if after_by_strategy:
        plot_retention(name, expert_acc, after_by_strategy, f"{WORK}/figures/retention_{TASK}_{name}.png")
    del expert
    if fisher_list is not None:
        del fisher_list, star_list
    torch.cuda.empty_cache()


# ===== CELL 7 — run every model (one failure logs + skips, never kills the run) =====
for name in MODELS_TO_RUN:
    try:
        run_model(name)
    except Exception as e:
        import traceback
        print(f"  [ERROR] model {name} FAILED — skipping: {e}", flush=True)
        traceback.print_exc()


# ===== CELL 8 — OPTIONAL joint upper bound (train on all tasks-so-far together) =====
# Off by default: joint is usually computed ONCE (over all 4 tasks) in the aggregation
# notebook. Enable RUN_JOINT to compute the per-transition joint ceiling here instead.
if RUN_JOINT:
    joint_rows = []
    rows = []
    for t in SEQ:
        rows += sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "train")
    joint_loader = sc._loader(rows, train_tf, BATCH_SIZE, True, WORKERS,
                              path_remap=REMAP, cache_dir=CACHE_DIR)
    print(f"[joint] {len(rows)} training images across {len(SEQ)} tasks")
    for name in MODELS_TO_RUN:
        try:
            t0 = time.time()
            model = sc.build_model(name, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
            model, _ = sc.train_naive(
                model, joint_loader, eval_loaders=ALL_VAL, optimizer=_adam(model), device=DEVICE,
                epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="joint",
                select_best=True, select_loaders=ALL_VAL, all_classes=cs.all_classes,
                seed=SEED, use_amp=USE_AMP)
            for t in SEQ:
                m = _eval(model, t)
                _record(name, "joint", t, m, round((time.time() - t0) / 60, 2) if t == TASK else None)
            print(f"  [joint] {name} done", flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"  [ERROR] joint {name} FAILED — skipping: {e}", flush=True)
            traceback.print_exc()


# ===== CELL 9 — write CSV / XLSX / DOCX / R-matrices and BUNDLE into one zip =====
def _safe(step, fn):
    try:
        fn(); print(f"  wrote {step}")
    except Exception as e:
        print(f"  [warn] {step} failed: {e}")

_safe("final_metrics.csv", lambda: sc.write_final_metrics_csv(final_rows, f"{WORK}/csv/final_metrics_{TASK}.csv"))
_safe("perclass.csv",      lambda: sc.write_perclass_csv(perclass_rows, f"{WORK}/csv/perclass_{TASK}.csv"))
_safe("clmetrics.csv",     lambda: sc.write_final_metrics_csv(clmetrics_rows, f"{WORK}/csv/clmetrics_{TASK}.csv"))
_safe("cl_matrices.json",  lambda: json.dump(
    {"seq": SEQ, "matrices": cl_matrices}, open(f"{WORK}/csv/cl_matrices_{TASK}.json", "w"), indent=2))
_safe("results.xlsx",      lambda: sc.build_results_xlsx(
    {"final_metrics": final_rows, "cl_metrics": clmetrics_rows, "per_class": perclass_rows},
    f"{WORK}/xlsx/{TASK}_results.xlsx"))
_safe("report.docx",       lambda: sc.build_report_docx(
    TASK, clmetrics_rows, f"{WORK}/docx/{TASK}_report.docx", figures_dir=f"{WORK}/figures",
    title=f"STARL v2 — {TASK} (continual-learning task)"))

run_manifest = {
    "task": TASK, "prev_task": PREV, "prior_tasks": PRIOR, "seq": SEQ,
    "seed": SEED, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
    "models": MODELS_TO_RUN,
    "strategies": [s for s, on in [("naive", RUN_NAIVE), ("rehearsal", RUN_REHEARSAL),
                   ("lwf", RUN_LWF), ("ewc", RUN_EWC), ("frozen_probe", RUN_FROZEN_PROBE),
                   ("independent", RUN_INDEPENDENT), ("joint", RUN_JOINT)] if on],
    "rehearsal_fraction": REHEARSAL_FRACTION, "reh_batch": REH_BATCH,
    "lwf": {"T": LWF_T, "lam": LWF_LAM}, "ewc_lam": EWC_LAM,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
zip_path = sc.bundle_outputs(TASK, WORK, out_zip=f"/kaggle/working/{TASK}_outputs.zip",
                             run_manifest=run_manifest)
print("\nBUNDLED ->", zip_path)
print(f"Download it, upload as a Kaggle dataset, and attach to the {SEQ[-1]} -> next-task notebook.")
import pandas as pd
if clmetrics_rows:
    print("\n=== CL METRICS (per model, per strategy) ===")
    print(pd.DataFrame(clmetrics_rows).to_string(index=False))
print("\n=== FINAL METRICS (head) ===")
print(pd.DataFrame(final_rows).to_string(index=False))
