# =============================================================================
# STARL — dump per-image predictions (enables bootstrap confidence intervals)
# =============================================================================
# WHY THIS IS STANDALONE RATHER THAN A PATCH TO starl_core.py
#   Bootstrapping resamples individual test images, so it needs one row per image
#   (true label, predicted label) — no run has ever saved that. The obvious fix is
#   to add two lines to starl_core.evaluate_on_task, but that would require
#   uploading a new version of the `starl-code` dataset, and the bias-correction and
#   task-order jobs are currently running against it. This notebook therefore does
#   its own inference and leaves starl_core untouched. Nothing that has already been
#   computed changes; this only adds a file the earlier runs never wrote.
#
# WHAT IT PRODUCES
#   predictions_{model}_{task}.csv  with columns: y_true, y_pred, correct, and the
#   predicted probability of both the true and the predicted class. The probabilities
#   also support a calibration analysis (ECE / reliability diagrams) later, which is
#   worth having for a clinical-deployment argument.
#
# Inference only: minutes, no training. Feed the output to Part B of
# STARL_significance.py.
#
# ATTACH: starl-code, starl-splits, all four image datasets, the T4 output zip(s).
# Internet ON only if 'kan' is in the model list (pykan).

# ===== CELL 1 — imports =====
import sys, os, glob, json
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import numpy as np, pandas as pd
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
for pkg, mod in (("pykan", "kan"),):
    try: __import__(mod)
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

# ===== CELL 2 — config =====
SEED, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 32, 224, 4
# The five single-seed architectures have no uncertainty estimate at all, so they
# matter most; the rest are included because the run is cheap.
MODELS_TO_RUN = ["resnet18", "resnet50", "resnet101", "resnet152", "vgg11", "vgg16",
                 "vgg19", "alexnet", "googlenet", "swin_tiny", "coatnet_0", "kan"]
WORK = "/kaggle/working/PREDICTIONS"
os.makedirs(f"{WORK}/csv", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs    = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
ALL   = cs.all_classes
print("device:", DEVICE)

# ===== CELL 3 — data =====
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

_, eval_tf = sc.build_transforms(IMAGE_SIZE)
TEST, PATHS = {}, {}
for t in TASKS:
    rows = sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), "test")
    PATHS[t] = [r[0] for r in rows]
    TEST[t] = sc._loader(rows, eval_tf, BATCH_SIZE, False, WORKERS,
                         path_remap=REMAP, cache_dir=CACHE_DIR)
    print(f"  {t}: {len(rows)} test images")

# ===== CELL 4 — inference with per-image output =====
@torch.no_grad()
def predict(model, loader):
    model.eval(); ys, ps, p_true, p_pred = [], [], [], []
    for images, labels in loader:
        out = model(images.to(DEVICE))
        prob = torch.softmax(out, 1).cpu().numpy()
        pred = prob.argmax(1)
        lab = labels.numpy()
        ys.extend(lab); ps.extend(pred)
        p_pred.extend(prob[np.arange(len(pred)), pred])
        # probability assigned to the TRUE class (needed for calibration work)
        p_true.extend(prob[np.arange(len(lab)), np.clip(lab, 0, prob.shape[1] - 1)])
    return np.array(ys), np.array(ps), np.array(p_true), np.array(p_pred)

def load_final(name):
    hits = glob.glob(f"/kaggle/input/**/expert_{TASKS[-1]}_{name}.pth", recursive=True)
    if not hits: return None
    m = sc.build_model(name, len(ALL), DEVICE, SEED, pretrained=False, feature_extract=True)
    m.load_state_dict(torch.load(hits[0], map_location="cpu", weights_only=False)["model_state_dict"])
    return m.to(DEVICE).eval()

summary = []
for name in MODELS_TO_RUN:
    try:
        model = load_final(name)
        if model is None:
            print(f"  [skip] {name}: no expert_{TASKS[-1]}_{name}.pth attached"); continue
        for t in TASKS:
            y, p, pt, pp = predict(model, TEST[t])
            df = pd.DataFrame({
                "image": [os.path.basename(x) for x in PATHS[t][:len(y)]],
                "y_true": y, "y_pred": p, "correct": (y == p).astype(int),
                "true_class": [ALL[i] if 0 <= i < len(ALL) else "?" for i in y],
                "pred_class": [ALL[i] if 0 <= i < len(ALL) else "?" for i in p],
                "prob_true": np.round(pt, 6), "prob_pred": np.round(pp, 6)})
            out = f"{WORK}/csv/predictions_{name}_{t}.csv"
            df.to_csv(out, index=False)
            acc = float((y == p).mean())
            summary.append({"model": name, "task": t, "n": len(y), "accuracy": acc})
            print(f"  {name:<11}{t:<11}n={len(y):<6}acc={acc:.4f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e:
        import traceback; print(f"  [ERROR] {name}: {e}"); traceback.print_exc()

# ===== CELL 5 — bundle =====
pd.DataFrame(summary).to_csv(f"{WORK}/csv/predictions_summary.csv", index=False)
zip_path = sc.bundle_outputs("PREDICTIONS", WORK, out_zip="/kaggle/working/PREDICTIONS_outputs.zip",
                             run_manifest={"experiment": "per_image_predictions",
                                           "models": MODELS_TO_RUN, "tasks": TASKS, "seed": SEED})
print("\nBUNDLED ->", zip_path)
print("\nSANITY CHECK: the accuracies above should match the corresponding cells of")
print("final_metrics_*.csv from the main runs. If they do not, the wrong checkpoint or")
print("the wrong test split has been loaded — stop and reconcile before using these files.")
if summary:
    print("\n" + pd.DataFrame(summary).pivot_table(index="model", columns="task",
                                                   values="accuracy").round(4).to_string())
print("\nNEXT: attach PREDICTIONS_outputs.zip to STARL_significance.py; Part B will "
      "detect the files automatically and emit bootstrap 95 % confidence intervals.")
