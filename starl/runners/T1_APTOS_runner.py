# =============================================================================
# STARL v2 — TASK 1 (APTOS base session) RUNNER
# =============================================================================
# HOW TO USE THIS FILE ON KAGGLE:
#   Each "# ===== CELL n =====" block below = ONE notebook cell. Copy each block
#   into its own cell, top to bottom, then Run All (or Save & Run All / Commit).
#
# ATTACH TO THE NOTEBOOK (right sidebar -> Add Input):
#   1. starl-code   — the dataset holding starl_core.py + starl_baselines.py
#   2. starl-splits — the Phase-1 output (unzipped)
#   3. the APTOS dataset — the SAME one used in Phase 1
# You do NOT set an APTOS path anywhere: the manifest already knows the image
# locations, and CELL 3 auto-resolves them no matter where APTOS is mounted.
#
# Task 1 is the BASE SESSION (no prior task to forget). It runs base training
# (-> the "expert" the next task loads) and the frozen-probe baseline.


# ===== CELL 1 — imports & attach the code library =====
import sys, os, json, time
CODE_DIR   = "/kaggle/input/starl-code"      # <-- folder with starl_core.py + starl_baselines.py
SPLITS_DIR = "/kaggle/input/starl-splits"    # <-- Phase-1 output (unzipped)
sys.path.append(CODE_DIR)
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
import starl_baselines as sb          # not used in Task 1, but confirms it imports
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
# python-docx is not preinstalled on Kaggle - install quietly (harmless if already present/offline)
try:
    import docx  # noqa
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"])


# ===== CELL 2 — config =====
TASK          = "T1_APTOS"
SEED          = 42
EPOCHS        = 15
LR            = 1e-4
BATCH_SIZE    = 32
IMAGE_SIZE    = 224
WORKERS       = 4
# Start with ONE model to validate the whole pipeline, then set the full list:
MODELS_TO_RUN = ["resnet50"]
# Full list once validated:
# MODELS_TO_RUN = ["resnet18","resnet50","resnet101","resnet152",
#                  "vgg11","vgg16","vgg19","alexnet","googlenet","swin_tiny","coatnet_0"]
RUN_FROZEN_PROBE = True   # set False for a quick base-only validation run

# WORK is the OUTPUT folder. It MUST be under /kaggle/working (writable).
# Do NOT point this at an input dataset (/kaggle/input/... is read-only).
WORK = "/kaggle/working/T1_APTOS"
for sub in ("models", "csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
# 224px image cache: built once on the first epoch, reused by every epoch AND every model
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print("device:", DEVICE, "| output ->", WORK)


# ===== CELL 3 — load class space, auto-resolve image paths, build Task-1 data =====
cs        = sc.load_class_space(SPLITS_DIR)
NUM       = cs.num_classes_through(TASK)          # 5 DR grades
TASK_IDX  = cs.task_idx(TASK)

# The manifest stores absolute image paths from Phase 1. If APTOS is mounted at a
# different path now, derive a one-time prefix swap so every path resolves.
def _index_inputs(root="/kaggle/input"):
    idx = {}
    for r, _d, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                idx.setdefault(f, os.path.join(r, f))
    return idx

def make_path_remap(splits_dir, task):
    rows = sc.read_manifest(sc.manifest_path(splits_dir, task), None)
    sample = rows[0][0]
    if os.path.exists(sample):
        return None, sample                        # paths already resolve — no remap
    found = _index_inputs().get(os.path.basename(sample))
    if not found:
        return None, None                          # file genuinely missing
    so, sf, k = sample.split("/"), found.split("/"), 0
    while k < min(len(so), len(sf)) and so[-1 - k] == sf[-1 - k]:
        k += 1
    old_pre, new_pre = "/".join(so[:len(so) - k]), "/".join(sf[:len(sf) - k])
    print(f"  path remap: '{old_pre}' -> '{new_pre}'")
    return (lambda p: p.replace(old_pre, new_pre, 1) if p.startswith(old_pre) else p), found

REMAP, sample = make_path_remap(SPLITS_DIR, TASK)
assert sample and os.path.exists(REMAP(sample) if REMAP else sample), (
    "APTOS images not found — attach the SAME APTOS dataset you used in Phase 1.")
print("  sample image resolves OK")

tr_loader, va_loader, te_loader = sc.build_task_loaders(
    SPLITS_DIR, TASK, BATCH_SIZE, IMAGE_SIZE, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
print(f"Task {TASK}: {NUM} classes -> {cs.task_classes[TASK]}")
print(f"  train {len(tr_loader.dataset)} | val {len(va_loader.dataset)} | test {len(te_loader.dataset)}")


# ===== CELL 4 — small plotting helper (confusion matrix) =====
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


# ===== CELL 5 — BASE training for each model (this becomes the Task-2 expert) =====
epoch_logger  = sc.MetricLogger(f"{WORK}/csv/epoch_history_{TASK}.csv")
final_rows    = []
perclass_rows = []

def _run_base(name):
    print("\n" + "=" * 70 + f"\n### BASE: {name}\n" + "=" * 70, flush=True)
    t0 = time.time()
    model = sc.build_model(name, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
    total_p, train_p = sc.count_params(model)
    print(f"  params: {total_p:,} total | {train_p:,} trainable")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    ckpt = f"{WORK}/models/expert_{TASK}_{name}.pth"     # <- the file Task 2 will load
    model, _hist = sc.train_naive(
        model, tr_loader, eval_loaders={TASK + "_val": va_loader}, optimizer=opt, device=DEVICE,
        epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="base",
        select_best=True,     # one eval pass/epoch (val); final TEST computed once below
        best_ckpt_path=ckpt, all_classes=cs.all_classes, seed=SEED)

    m = sc.evaluate_on_task(model, te_loader, TASK_IDX, cs.all_classes, DEVICE)
    final_rows.append({"model": name, "experiment": "base", "task": TASK,
                       "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                       "f1_macro": m["f1_macro"], "precision_weighted": m["precision_weighted"],
                       "recall_weighted": m["recall_weighted"],
                       "roc_auc": m["roc_auc_ovr_weighted"],
                       "train_minutes": round((time.time() - t0) / 60, 2)})
    for cls, v in m["per_class"].items():
        perclass_rows.append({"model": name, "task": TASK, "class": cls, **v})
    plot_confusion(m["confusion_matrix"], m["class_names"], f"{name} — {TASK} (test)",
                   f"{WORK}/figures/cm_{TASK}_{name}.png")
    print(f"  ✓ {name} BASE done — test acc {m['accuracy']:.4f} | F1w {m['f1_weighted']:.4f}",
          flush=True)


for name in MODELS_TO_RUN:
    try:
        _run_base(name)
    except Exception as e:
        import traceback
        print(f"  [ERROR] base {name} FAILED - skipping: {e}", flush=True)
        traceback.print_exc()


# ===== CELL 6 — FROZEN-PROBE baseline (backbone frozen, only the head trains) =====
def _run_probe(name):
    print("\n" + "-" * 70 + f"\n### FROZEN-PROBE: {name}\n" + "-" * 70, flush=True)
    model = sc.build_model(name, NUM, DEVICE, SEED, pretrained=True,
                           feature_extract=True, frozen_probe=True)
    _, train_p = sc.count_params(model)
    print(f"  trainable params (head only): {train_p:,}")
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    model, _ = sc.train_naive(
        model, tr_loader, eval_loaders={TASK + "_val": va_loader}, optimizer=opt, device=DEVICE,
        epochs=EPOCHS, model_name=name, logger=epoch_logger, experiment="frozen_probe",
        select_best=True, all_classes=cs.all_classes, seed=SEED)
    m = sc.evaluate_on_task(model, te_loader, TASK_IDX, cs.all_classes, DEVICE)
    final_rows.append({"model": name, "experiment": "frozen_probe", "task": TASK,
                       "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                       "f1_macro": m["f1_macro"], "precision_weighted": m["precision_weighted"],
                       "recall_weighted": m["recall_weighted"],
                       "roc_auc": m["roc_auc_ovr_weighted"], "train_minutes": None})
    print(f"  ✓ {name} FROZEN-PROBE done — test acc {m['accuracy']:.4f}", flush=True)


for name in (MODELS_TO_RUN if RUN_FROZEN_PROBE else []):
    try:
        _run_probe(name)
    except Exception as e:
        import traceback
        print(f"  [ERROR] frozen-probe {name} FAILED - skipping: {e}", flush=True)
        traceback.print_exc()


# ===== CELL 7 — write CSV / XLSX / DOCX and BUNDLE everything into one zip =====
def _safe(step, fn):
    try:
        fn(); print(f"  wrote {step}")
    except Exception as e:
        print(f"  [warn] {step} failed: {e}")

_safe("final_metrics.csv", lambda: sc.write_final_metrics_csv(final_rows, f"{WORK}/csv/final_metrics_{TASK}.csv"))
_safe("perclass.csv",      lambda: sc.write_perclass_csv(perclass_rows, f"{WORK}/csv/perclass_{TASK}.csv"))
_safe("results.xlsx",      lambda: sc.build_results_xlsx({"final_metrics": final_rows, "per_class": perclass_rows}, f"{WORK}/xlsx/{TASK}_results.xlsx"))
_safe("report.docx",       lambda: sc.build_report_docx(TASK, final_rows, f"{WORK}/docx/{TASK}_report.docx", figures_dir=f"{WORK}/figures", title=f"STARL v2 - {TASK} (APTOS base)"))

run_manifest = {"task": TASK, "seed": SEED, "epochs": EPOCHS, "lr": LR,
                "models": MODELS_TO_RUN, "device": str(DEVICE),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
zip_path = sc.bundle_outputs(TASK, WORK, out_zip="/kaggle/working/T1_APTOS_outputs.zip",
                             run_manifest=run_manifest)
print("\nBUNDLED ->", zip_path)
print("Download it from the Output tab, then upload as a Kaggle dataset for Task 2.")
import pandas as pd
print("\n=== FINAL METRICS ===")
print(pd.DataFrame(final_rows).to_string(index=False))
