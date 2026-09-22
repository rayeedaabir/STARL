# =============================================================================
# STARL v2 — JOINT / OFFLINE UPPER BOUND  (professor #7)
# =============================================================================
# Trains each model ONCE on ALL four tasks' training data combined (the full
# 19-class space) and evaluates on every task's frozen test split. This is the
# "no forgetting because it saw everything at once" ceiling the CL methods are
# measured against. NOT sequential -> no BWT/FM, just the per-task upper bound.
#
# Scope now = the 4 representative models (matches the multi-seed subset). To do
# all 12 later, just extend MODELS_TO_RUN (kan joint is very heavy — resume-run it
# with the KAN runner's machinery if the professor asks).
#
# ATTACH (Add Input): starl-code, starl-splits, and ALL FOUR image datasets
# (APTOS + ODIR + LAG + HAM). No expert zips — joint trains from ImageNet.


# ===== CELL 1 — imports & attach the code library =====
import sys, os, json, time, glob
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
try: import docx  # noqa
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"])


# ===== CELL 2 — config =====
SEED, EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 15, 1e-4, 32, 224, 4
USE_AMP = True
# The 4 representative models (extend to the full 12 if the professor asks for it):
MODELS_TO_RUN = ["resnet50", "resnet152", "swin_tiny", "coatnet_0"]
# MODELS_TO_RUN = ["resnet18","resnet50","resnet101","resnet152","vgg11","vgg16","vgg19",
#                  "alexnet","googlenet","swin_tiny","coatnet_0"]   # 'kan' = separate heavy run

WORK = "/kaggle/working/JOINT"
for sub in ("models", "csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print("device:", DEVICE, "| output ->", WORK)


# ===== CELL 3 — class space, path resolver, joint train loader + per-task eval loaders =====
cs   = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)                       # [T1_APTOS, T2_ODIR, T3_LAG, T4_HAM]
NUM  = len(cs.all_classes)                        # full 19-class head
print("joint over tasks:", TASKS, "| classes:", NUM)

_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_INPUT_IDX = None
def _build_input_index(root="/kaggle/input"):
    idx = {}
    for r, _d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(_IMG_EXT):
                idx.setdefault(f, os.path.join(r, f))
    return idx
def REMAP(path):
    if os.path.exists(path): return path
    global _INPUT_IDX
    if _INPUT_IDX is None:
        print("  building /kaggle/input image index (first miss)...", flush=True)
        _INPUT_IDX = _build_input_index(); print(f"  indexed {len(_INPUT_IDX)} images", flush=True)
    return _INPUT_IDX.get(os.path.basename(path), path)

for t in TASKS:
    _p0 = sc.read_manifest(sc.manifest_path(SPLITS_DIR, t))[0][0]
    assert os.path.exists(REMAP(_p0)), f"{t}: images not found — attach the SAME dataset used in Phase 1."
print("  all task images resolve OK")

train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
# joint TRAIN = concatenated train splits of all tasks (cached + remapped)
joint_rows = []
for t in TASKS:
    joint_rows += sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "train")
joint_loader = sc._loader(joint_rows, train_tf, BATCH_SIZE, True, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
# per-task val (selection) and test (final eval)
ALL_VAL, ALL_TEST = {}, {}
for t in TASKS:
    ALL_VAL[t]  = sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "val"),
                             eval_tf, BATCH_SIZE, False, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
    ALL_TEST[t] = sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "test"),
                             eval_tf, BATCH_SIZE, False, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
print(f"  joint train images: {len(joint_rows)}")


# ===== CELL 4 — confusion-matrix helper =====
def plot_confusion(cm, class_names, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.4 + 0.6 * len(class_names), 1.2 + 0.6 * len(class_names)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# ===== CELL 5 — train each model jointly, evaluate on every task's test set =====
epoch_logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_JOINT.csv")
final_rows, perclass_rows, summary_rows = [], [], []

def _run_joint(name):
    print("\n" + "=" * 70 + f"\n### JOINT: {name}\n" + "=" * 70, flush=True)
    t0 = time.time()
    model = sc.build_model(name, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    ckpt = f"{WORK}/models/joint_{name}.pth"
    model, _ = sc.train_naive(
        model, joint_loader, eval_loaders=ALL_VAL, optimizer=opt, device=DEVICE, epochs=EPOCHS,
        model_name=name, logger=epoch_logger, experiment="joint", select_best=True,
        select_loaders=None, best_ckpt_path=ckpt, all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
    accs = []
    for t in TASKS:
        m = sc.evaluate_on_task(model, ALL_TEST[t], cs.task_idx(t), cs.all_classes, DEVICE)
        accs.append(m["accuracy"])
        final_rows.append({"model": name, "experiment": "joint", "eval_task": t,
                           "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                           "f1_macro": m["f1_macro"], "precision_weighted": m["precision_weighted"],
                           "recall_weighted": m["recall_weighted"], "roc_auc": m["roc_auc_ovr_weighted"],
                           "train_minutes": round((time.time() - t0) / 60, 2) if t == TASKS[-1] else None})
        for cls, v in m["per_class"].items():
            perclass_rows.append({"model": name, "experiment": "joint", "eval_task": t, "class": cls, **v})
        plot_confusion(m["confusion_matrix"], m["class_names"], f"{name} — {t} joint (test)",
                       f"{WORK}/figures/cm_JOINT_{name}_{t}.png")
        print(f"  {t}: acc {m['accuracy']:.4f}", flush=True)
    summary_rows.append({"model": name, "experiment": "joint", "joint_avg_accuracy": float(np.mean(accs))})
    print(f"  ✓ {name} joint avg acc {np.mean(accs):.4f}", flush=True)

for name in MODELS_TO_RUN:
    try:
        _run_joint(name)
    except Exception as e:
        import traceback; print(f"  [ERROR] joint {name} FAILED — skipping: {e}", flush=True); traceback.print_exc()


# ===== CELL 6 — write outputs + bundle =====
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")

_safe("final_metrics_JOINT.csv", lambda: sc.write_final_metrics_csv(final_rows, f"{WORK}/csv/final_metrics_JOINT.csv"))
_safe("perclass_JOINT.csv",      lambda: sc.write_perclass_csv(perclass_rows, f"{WORK}/csv/perclass_JOINT.csv"))
_safe("joint_summary.csv",       lambda: sc.write_final_metrics_csv(summary_rows, f"{WORK}/csv/joint_summary_JOINT.csv"))
_safe("results.xlsx",            lambda: sc.build_results_xlsx(
    {"joint_final_metrics": final_rows, "joint_summary": summary_rows, "per_class": perclass_rows},
    f"{WORK}/xlsx/JOINT_results.xlsx"))
_safe("report.docx",             lambda: sc.build_report_docx("JOINT", summary_rows, f"{WORK}/docx/JOINT_report.docx",
                                                              figures_dir=f"{WORK}/figures", title="STARL v2 — Joint upper bound"))

run_manifest = {"experiment": "joint", "tasks": TASKS, "num_classes": NUM, "seed": SEED,
                "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "models": MODELS_TO_RUN,
                "joint_train_images": len(joint_rows), "device": str(DEVICE),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
zip_path = sc.bundle_outputs("JOINT", WORK, out_zip="/kaggle/working/JOINT_outputs.zip", run_manifest=run_manifest)
print("\nBUNDLED ->", zip_path)
print("Upload as a dataset and attach to the aggregation notebook — it becomes the upper-bound row.")
import pandas as pd
if summary_rows:
    print("\n=== JOINT UPPER BOUND (avg over tasks) ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))
if final_rows:
    print("\n=== per-task joint accuracy ===")
    print(pd.DataFrame(final_rows)[["model", "eval_task", "accuracy", "f1_weighted"]].to_string(index=False))
