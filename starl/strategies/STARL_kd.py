# =============================================================================
# STARL v2 — KNOWLEDGE DISTILLATION  (compression preserves CL retention)
# =============================================================================
# Distils the FINAL continual ResNet-50 (the rehearsal T4 model, 19-class) into a
# much smaller MobileNet-V2 student, then evaluates the student on every task's
# frozen test split. Framing for the paper: "the deployed continual model can be
# compressed ~7x for edge/hospital devices WITHOUT losing its retention" — the
# honest, de-leaked version of the old KD result.
#
# ATTACH (Add Input): starl-code, starl-splits, all FOUR image datasets
# (APTOS+ODIR+LAG+HAM), and the T4 CNN output zip that holds
# expert_T4_HAM_resnet50.pth (the teacher). Internet ON (MobileNet-V2 weights).


# ===== CELL 1 — imports & attach the code library =====
import sys, os, json, time, glob, copy
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
try: import docx  # noqa
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx"])


# ===== CELL 2 — config =====
TEACHER_NAME = "resnet50"           # the deployed final CL model (rehearsal T4)
TEACHER_CKPT = None                 # None = auto-find expert_T4_HAM_resnet50.pth
SEED, EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 15, 1e-4, 32, 224, 4
USE_AMP = True
KD_T = 4.0                          # distillation temperature
# loss = KD_ALPHA*CE(true labels) + (1-KD_ALPHA)*T^2*KL(student||teacher).
# KEEP THIS ~0 (PURE distillation). With alpha>0 the student also gets TRUE labels on
# ALL tasks' data at once, so it becomes a JOINT-trained model and beats the sequential
# teacher — that is the joint-vs-CL gap, NOT "MobileNetV2 > ResNet-50". Pure distillation
# makes the student REPLICATE the teacher, which is the honest "compression preserves
# retention" test (expect student ~= teacher). See STARL_kd notes / PROGRESS.md.
KD_ALPHA = 0.0

WORK = "/kaggle/working/KD"
for sub in ("models", "csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print("device:", DEVICE, "| teacher:", TEACHER_NAME, "-> student: mobilenet_v2")


# ===== CELL 3 — class space, path resolver, combined-train + per-task eval loaders =====
cs    = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
NUM   = len(cs.all_classes)                     # 19
print("tasks:", TASKS, "| classes:", NUM)

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
kd_rows = []
for t in TASKS:
    kd_rows += sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "train")
kd_loader = sc._loader(kd_rows, train_tf, BATCH_SIZE, True, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
ALL_VAL, ALL_TEST = {}, {}
for t in TASKS:
    ALL_VAL[t]  = sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "val"),
                             eval_tf, BATCH_SIZE, False, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
    ALL_TEST[t] = sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "test"),
                             eval_tf, BATCH_SIZE, False, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
print(f"  KD train images (all tasks): {len(kd_rows)}")


# ===== CELL 4 — load the teacher (final CL ResNet-50) + build the MobileNet-V2 student =====
if TEACHER_CKPT is None:
    _h = glob.glob(f"/kaggle/input/**/expert_T4_HAM_{TEACHER_NAME}.pth", recursive=True)
    TEACHER_CKPT = _h[0] if _h else None
assert TEACHER_CKPT and os.path.exists(TEACHER_CKPT), (
    f"Teacher checkpoint expert_T4_HAM_{TEACHER_NAME}.pth not found — attach the T4 CNN output zip.")
print("  teacher ckpt:", TEACHER_CKPT)

teacher = sc.build_model(TEACHER_NAME, NUM, DEVICE, SEED, pretrained=False, feature_extract=True)
_ck = torch.load(TEACHER_CKPT, map_location="cpu", weights_only=False)     # our own file
teacher.load_state_dict(_ck["model_state_dict"]); teacher.to(DEVICE).eval()
for p in teacher.parameters(): p.requires_grad = False

def build_student():
    m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)   # ImageNet init (needs internet)
    m.classifier[1] = nn.Linear(m.last_channel, NUM)                        # 1280 -> 19
    return m.to(DEVICE)                                                     # full fine-tuning (small net)

student = build_student()
tp, _ = sc.count_params(teacher); spar, _ = sc.count_params(student)
print(f"  teacher params {tp:,} | student params {spar:,} | compression {tp/spar:.1f}x")


# ===== CELL 5 — distillation training (KD loss), select best student on mean val =====
def evaluate_all(model, loaders):
    return {t: sc.quick_accuracy(model, ld, DEVICE) for t, ld in loaders.items()}

epoch_logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_KD.csv")
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=LR)
use_amp = USE_AMP and DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
best_avg, best_state = -1.0, None

for epoch in range(1, EPOCHS + 1):
    t0 = time.time(); student.train(); run = correct = total = 0.0
    for images, labels in kd_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            s_out = student(images)
            with torch.no_grad():
                t_out = teacher(images)
            ce = criterion(s_out, labels)
            kd = F.kl_div(F.log_softmax(s_out / KD_T, dim=1), F.softmax(t_out / KD_T, dim=1),
                          reduction="batchmean") * (KD_T * KD_T)
            loss = KD_ALPHA * ce + (1.0 - KD_ALPHA) * kd
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        run += loss.item() * images.size(0)
        _, pr = torch.max(s_out, 1); correct += (pr == labels).sum().item(); total += images.size(0)
    val = evaluate_all(student, ALL_VAL); avg = float(np.mean(list(val.values())))
    row = {"model": "mobilenet_v2", "experiment": "kd", "epoch": epoch,
           "time_s": round(time.time() - t0, 2), "train_loss": run / total, "train_acc": correct / total}
    row.update({f"{t}_val_acc": val[t] for t in TASKS})
    epoch_logger.log_epoch(row)
    print(f"  [kd] ep {epoch}/{EPOCHS} {row['time_s']:.0f}s loss {run/total:.4f} | mean-val {avg:.4f} | "
          + " ".join(f"{t}:{val[t]:.3f}" for t in TASKS), flush=True)
    if avg > best_avg:
        best_avg, best_state = avg, copy.deepcopy(student.state_dict())
        sc.save_checkpoint(student, f"{WORK}/models/kd_mobilenet_v2.pth", cs.all_classes, seed=SEED,
                           extra={"epoch": epoch, "avg_acc": avg, "teacher": TEACHER_NAME})
if best_state is not None:
    student.load_state_dict(best_state)


# ===== CELL 6 — evaluate teacher vs student on every task's test set =====
def plot_confusion(cm, class_names, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.4 + 0.6 * len(class_names), 1.2 + 0.6 * len(class_names)))
    ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

final_rows, perclass_rows = [], []
def _record(who, model_obj):
    accs = []
    for t in TASKS:
        m = sc.evaluate_on_task(model_obj, ALL_TEST[t], cs.task_idx(t), cs.all_classes, DEVICE)
        accs.append(m["accuracy"])
        final_rows.append({"model": who, "experiment": "kd", "eval_task": t, "accuracy": m["accuracy"],
                           "f1_weighted": m["f1_weighted"], "f1_macro": m["f1_macro"],
                           "precision_weighted": m["precision_weighted"], "recall_weighted": m["recall_weighted"],
                           "roc_auc": m["roc_auc_ovr_weighted"], "train_minutes": None})
        for cls, v in m["per_class"].items():
            perclass_rows.append({"model": who, "experiment": "kd", "eval_task": t, "class": cls, **v})
        if who == "kd_student_mobilenet_v2":
            plot_confusion(m["confusion_matrix"], m["class_names"], f"KD student — {t} (test)",
                           f"{WORK}/figures/cm_KD_student_{t}.png")
        print(f"  [{who}] {t}: acc {m['accuracy']:.4f}", flush=True)
    return float(np.mean(accs))

teacher_acc = _record("kd_teacher_resnet50", teacher)
student_acc = _record("kd_student_mobilenet_v2", student)
summary_rows = [{"role": "teacher (resnet50)", "avg_accuracy": teacher_acc, "params": tp},
                {"role": "student (mobilenet_v2)", "avg_accuracy": student_acc, "params": spar,
                 "compression_x": round(tp / spar, 2), "retention_gap": round(teacher_acc - student_acc, 4)}]
print(f"\n  teacher avg {teacher_acc:.4f} | student avg {student_acc:.4f} | "
      f"gap {teacher_acc-student_acc:.4f} | {tp/spar:.1f}x smaller", flush=True)


# ===== CELL 7 — write outputs + bundle =====
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")

_safe("final_metrics_KD.csv", lambda: sc.write_final_metrics_csv(final_rows, f"{WORK}/csv/final_metrics_KD.csv"))
_safe("perclass_KD.csv",      lambda: sc.write_perclass_csv(perclass_rows, f"{WORK}/csv/perclass_KD.csv"))
_safe("kd_summary.csv",       lambda: sc.write_final_metrics_csv(summary_rows, f"{WORK}/csv/kd_summary_KD.csv"))
_safe("results.xlsx",         lambda: sc.build_results_xlsx(
    {"kd_summary": summary_rows, "kd_final_metrics": final_rows, "per_class": perclass_rows},
    f"{WORK}/xlsx/KD_results.xlsx"))
_safe("report.docx",          lambda: sc.build_report_docx("KD", summary_rows, f"{WORK}/docx/KD_report.docx",
                                                           figures_dir=f"{WORK}/figures",
                                                           title="STARL v2 — Knowledge Distillation (compression preserves retention)"))

run_manifest = {"experiment": "kd", "teacher": TEACHER_NAME, "student": "mobilenet_v2", "tasks": TASKS,
                "num_classes": NUM, "seed": SEED, "epochs": EPOCHS, "kd_T": KD_T, "kd_alpha": KD_ALPHA,
                "teacher_params": tp, "student_params": spar, "compression_x": round(tp / spar, 2),
                "device": str(DEVICE)}
zip_path = sc.bundle_outputs("KD", WORK, out_zip="/kaggle/working/KD_outputs.zip", run_manifest=run_manifest)
print("\nBUNDLED ->", zip_path)
import pandas as pd
print("\n=== KD SUMMARY ===")
print(pd.DataFrame(summary_rows).to_string(index=False))
print("\n=== per-task: teacher vs student ===")
print(pd.DataFrame(final_rows).pivot_table(index="eval_task", columns="model", values="accuracy").round(4).to_string())
