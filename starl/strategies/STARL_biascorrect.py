# =============================================================================
# STARL — CLASSIFIER-BIAS DIAGNOSIS AND POST-HOC CORRECTION
# =============================================================================
# WHY: naive/LwF/EWC drive prior-task accuracy to ~0, yet CKA shows their
# representations of those tasks are largely intact. The hypothesis is that the
# failure lives in the FINAL LAYER: new-class logits receive abundant positive
# gradient while old-class logits receive none, so argmax over the full label
# space almost never selects an old class. If that is right, a correction applied
# to the last layer alone — with NO retraining and NO stored data — should recover
# a substantial part of the lost accuracy.
#
# This converts a negative result ("the baselines failed") into a diagnosis plus a
# remedy, which is a methodological contribution rather than an absence.
#
# WHAT IT DOES
#   1. Runs the naive chain T2 -> T3 -> T4 from the existing T1 expert and keeps
#      the final model (the main runs never saved these).
#   2. Applies three post-hoc corrections to the final linear layer and re-evaluates
#      on every task's frozen test split:
#        raw          — no correction (the number reported in the paper)
#        WA           — weight alignment: rescale new-class weight vectors so their
#                       mean norm matches the old-class mean norm (Zhao et al., 2020)
#        WA+nobias    — weight alignment and the final bias vector zeroed
#   3. Reports accuracy per task before and after, plus ACC/BWT/FM.
#
# COST: ~25 min per fast model (3 tasks x 15 epochs). Defaults to three models
# spanning three families; ~2 h total. No new data is used at any point.
#
# ATTACH: starl-code, starl-splits, all four image datasets, the T1 output zip.


# ===== CELL 1 — imports =====
import sys, os, glob, json, time, copy
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, torch.nn as nn
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# ===== CELL 2 — config =====
SEED, EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 15, 1e-4, 32, 224, 4
USE_AMP = True
# fast, and spanning three families. KAN is excluded: its spline head has no final
# linear layer to rescale, so the correction is undefined for it.
MODELS_TO_RUN = ["resnet18", "googlenet", "swin_tiny"]
STRATEGIES    = ["naive"]        # add "ewc"/"lwf" to correct those too (each ~doubles the time)

WORK = "/kaggle/working/BIASCORRECT"
for sub in ("csv", "xlsx", "figures", "models"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs    = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
print("device:", DEVICE, "| models:", MODELS_TO_RUN)


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

train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def rows(t, s): return sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), s)
def mk(r, tf, b, sh): return sc._loader(r, tf, b, sh, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
for t in TASKS:
    assert os.path.exists(REMAP(rows(t, "train")[0][0])), f"{t}: images missing"
TRAIN = {t: mk(rows(t, "train"), train_tf, BATCH_SIZE, True)  for t in TASKS}
VAL   = {t: mk(rows(t, "val"),   eval_tf,  BATCH_SIZE, False) for t in TASKS}
TEST  = {t: mk(rows(t, "test"),  eval_tf,  BATCH_SIZE, False) for t in TASKS}
print("  loaders ready")


# ===== CELL 4 — locate the final linear layer of any architecture =====
def final_linear(model, name):
    """Return the last nn.Linear of the classifier head — the layer whose rows are
    the per-class weight vectors. Our heads are:
      resnet/googlenet : model.fc        = Sequential(Linear, ReLU, Dropout, Linear)
      vgg/alexnet      : model.classifier[6] = Linear
      transformers     : model.head      = Sequential(Linear, ReLU, Dropout, Linear)"""
    head = sc._head_module(model, name)
    if isinstance(head, nn.Linear): return head
    lin = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if not lin: raise ValueError(f"no Linear layer in the head of {name}")
    return lin[-1]

@torch.no_grad()
def weight_align(model, name, n_old, zero_bias=False):
    """Post-hoc correction. Rescales the weight vectors of the classes introduced at
    the final task so that their mean L2 norm equals that of the older classes.
    Returns a corrected COPY; the original is untouched."""
    m = copy.deepcopy(model)
    fc = final_linear(m, name)
    W = fc.weight.data
    if n_old <= 0 or n_old >= W.shape[0]:
        return m, float("nan")
    old_norm = W[:n_old].norm(dim=1).mean()
    new_norm = W[n_old:].norm(dim=1).mean()
    gamma = float(old_norm / (new_norm + 1e-12))
    W[n_old:] *= gamma
    if zero_bias and fc.bias is not None:
        fc.bias.data.zero_()
    return m, gamma


# ===== CELL 5 — run the naive chain and keep the final model =====
def load_expert(name, task):
    hits = glob.glob(f"/kaggle/input/**/expert_{task}_{name}.pth", recursive=True)
    if not hits: return None
    m = sc.build_model(name, cs.num_classes_through(task), DEVICE, SEED,
                       pretrained=False, feature_extract=True)
    m.load_state_dict(torch.load(hits[0], map_location="cpu", weights_only=False)["model_state_dict"])
    return m.to(DEVICE).eval()

logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_biascorrect.csv")

def run_chain(name, strategy):
    """Train the chain T2->T3->T4 under `strategy`, branching each step from the
    maintained model exactly as in the main experiments, and return the final model."""
    model = load_expert(name, TASKS[0])
    assert model is not None, f"attach the T1 zip — expert_{TASKS[0]}_{name}.pth not found"
    for ti in range(1, len(TASKS)):
        task = TASKS[ti]
        student = sc.build_model(name, cs.num_classes_through(task), DEVICE, SEED,
                                 pretrained=False, feature_extract=True)
        student, _ = sc.transfer_backbone(model, student, name)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=LR)
        seen = TASKS[:ti + 1]
        student, _ = sc.train_naive(student, TRAIN[task], eval_loaders={t: VAL[t] for t in seen},
                                    optimizer=opt, device=DEVICE, epochs=EPOCHS,
                                    model_name=f"{name}_{strategy}", logger=logger,
                                    experiment=strategy, select_best=True,
                                    select_loaders={task: VAL[task]},
                                    all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        del model; model = student
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return model

def eval_all(model):
    return {t: sc.evaluate_on_task(model, TEST[t], cs.task_idx(t), cs.all_classes, DEVICE)["accuracy"]
            for t in TASKS}

def cl_metrics(final_acc, diag):
    N = len(TASKS)
    acc = float(np.mean([final_acc[t] for t in TASKS]))
    bwt = float(np.mean([final_acc[t] - diag[t] for t in TASKS[:-1] if t in diag]))
    return acc, bwt

rows_out = []
for name in MODELS_TO_RUN:
    for strat in STRATEGIES:
        print("\n" + "=" * 74 + f"\n### {name} / {strat}\n" + "=" * 74, flush=True)
        t0 = time.time()
        try:
            model = run_chain(name, strat)
            n_old = cs.num_classes_through(TASKS[-2])      # classes before the final task
            variants = {"raw": model}
            wa, gamma = weight_align(model, name, n_old, zero_bias=False)
            variants["WA"] = wa
            wab, _ = weight_align(model, name, n_old, zero_bias=True)
            variants["WA+nobias"] = wab
            print(f"  weight-alignment factor gamma = {gamma:.4f} "
                  f"(new-class weights were {1/gamma:.2f}x larger than old)")
            for vname, vm in variants.items():
                a = eval_all(vm)
                rows_out.append({"model": name, "strategy": strat, "correction": vname,
                                 **{f"acc_{t}": a[t] for t in TASKS},
                                 "ACC": float(np.mean(list(a.values()))),
                                 "gamma": gamma if vname != "raw" else None,
                                 "minutes": round((time.time() - t0) / 60, 1)})
                print(f"  {vname:<10} " + "  ".join(f"{t}:{a[t]:.3f}" for t in TASKS)
                      + f"  | ACC {np.mean(list(a.values())):.4f}", flush=True)
            sc.save_checkpoint(model, f"{WORK}/models/{strat}_final_{name}.pth",
                               cs.all_classes, seed=SEED, extra={"strategy": strat})
            del model, variants
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}/{strat}: {e}"); traceback.print_exc()


# ===== CELL 6 — outputs =====
df = pd.DataFrame(rows_out)
def _safe(s, f):
    try: f(); print("  wrote", s)
    except Exception as e: print(f"  [warn] {s}: {e}")
_safe("biascorrect_results.csv", lambda: df.to_csv(f"{WORK}/csv/biascorrect_results.csv", index=False))
_safe("results.xlsx", lambda: sc.build_results_xlsx({"bias_correction": rows_out},
                                                    f"{WORK}/xlsx/BIASCORRECT_results.xlsx"))

def fig():
    if df.empty: return
    prior = [t for t in TASKS[:-1]]
    fig, ax = plt.subplots(figsize=(2 + 1.8 * len(MODELS_TO_RUN), 4.6))
    labs, x = [], 0; ticks = []
    for name in df.model.unique():
        d = df[df.model == name]
        for k, corr in enumerate(["raw", "WA", "WA+nobias"]):
            r = d[d.correction == corr]
            if r.empty: continue
            v = float(np.mean([r.iloc[0][f"acc_{t}"] for t in prior]))
            ax.bar(x + k * 0.25, v, 0.25, color=["#c44", "#4a7ebb", "#3d8b52"][k],
                   label=corr if name == df.model.unique()[0] else None)
        ticks.append(x + 0.25); labs.append(name); x += 1.2
    ax.set_xticks(ticks); ax.set_xticklabels(labs)
    ax.set_ylabel("mean accuracy on prior tasks"); ax.set_ylim(0, 1)
    ax.set_title("Post-hoc correction of the final layer recovers prior-task accuracy")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/bias_correction.png", dpi=300); plt.close(fig)
    print("  fig bias_correction")
_safe("figure", fig)

zip_path = sc.bundle_outputs("BIASCORRECT", WORK, out_zip="/kaggle/working/BIASCORRECT_outputs.zip",
                             run_manifest={"experiment": "bias_correction", "models": MODELS_TO_RUN,
                                           "strategies": STRATEGIES, "seed": SEED, "epochs": EPOCHS})
print("\nBUNDLED ->", zip_path)
if not df.empty:
    print("\n=== BIAS CORRECTION ===")
    print(df[["model", "strategy", "correction"] + [f"acc_{t}" for t in TASKS] + ["ACC", "gamma"]]
          .round(4).to_string(index=False))
    print("\nHOW TO READ: compare the 'raw' row with 'WA' for the same model. If the prior-task")
    print("columns rise substantially while the final-task column barely moves, the collapse was")
    print("a property of the classifier's output scale and not a loss of the representation.")
