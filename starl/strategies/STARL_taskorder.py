# =============================================================================
# STARL — TASK-ORDER ABLATION  (direct test of the representational-proximity claim)
# =============================================================================
# THE HYPOTHESIS (Section 4.2): interference between two tasks is governed by how
# similar their representations are, not by how many classes the later task adds.
# Evidence so far: adding +7 RETINAL classes to a retinal model retained 0.299 of
# the previous task, whereas adding +7 DERMOSCOPIC classes retained 0.866.
#
# THE PREDICTION THIS TESTS. In the original order the dissimilar task (skin) comes
# LAST, so its low-interference transition is the final one and the three fundus
# tasks are learned consecutively at the start. If we move the skin task to the
# FRONT, then every remaining transition is within-modality, and the hypothesis
# predicts MORE total forgetting — lower final ACC and more negative BWT — even
# though exactly the same four datasets are learned.
#
# If instead the two orders give similar results, the proximity account is wrong or
# incomplete and Section 4.2 must be softened. Either outcome is informative, which
# is what makes this worth running.
#
# CONTROLLED COMPARISON. Reordering changes how the head would grow (5→12→12→19
# originally, 7→12→19→19 with skin first), which would confound the comparison. We
# therefore use a FIXED 19-wide head for BOTH orders and re-run the original order
# under that setting, so the ONLY difference between the two arms is the order.
# Class indices are unchanged throughout; classes not yet seen simply receive no
# positive gradient.
#
# COST: 2 orders x 2 models x 4 tasks x 15 epochs ~ 2 h on fast models.
# ATTACH: starl-code, starl-splits, all four image datasets. No expert zip needed
# (each arm trains its own first task, since the first task differs between arms).


# ===== CELL 1 — imports =====
import sys, os, glob, json, time, copy
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# ===== CELL 2 — config =====
SEED, EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 15, 1e-4, 32, 224, 4
USE_AMP = True
MODELS_TO_RUN = ["resnet18", "googlenet"]      # fast; resnet18 is the reference model
REHEARSAL_FRACTION, REH_BATCH = 0.10, 4        # identical to the main experiments
SESSION_MAX_MINUTES = 660

WORK = "/kaggle/working/TASKORDER"
for sub in ("csv", "xlsx", "figures", "models"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs = sc.load_class_space(SPLITS_DIR)
NUM_ALL = len(cs.all_classes)                  # fixed 19-wide head for both arms

ORDERS = {
  "original":  ["T1_APTOS", "T2_ODIR", "T3_LAG", "T4_HAM"],   # dissimilar task LAST
  "skin_first": ["T4_HAM", "T1_APTOS", "T2_ODIR", "T3_LAG"],  # dissimilar task FIRST
}
print("device:", DEVICE, "| fixed head:", NUM_ALL, "classes")
for k, v in ORDERS.items(): print(f"  {k:<11} {' -> '.join(v)}")


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
def rws(t, s): return sc.read_manifest(sc.manifest_path(SPLITS_DIR, t), s)
def mk(r, tf, b, sh): return sc._loader(r, tf, b, sh, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
ALLT = list(cs.task_order)
for t in ALLT:
    assert os.path.exists(REMAP(rws(t, "train")[0][0])), f"{t}: images missing"
TRAIN = {t: mk(rws(t, "train"), train_tf, BATCH_SIZE, True)  for t in ALLT}
VAL   = {t: mk(rws(t, "val"),   eval_tf,  BATCH_SIZE, False) for t in ALLT}
TEST  = {t: mk(rws(t, "test"),  eval_tf,  BATCH_SIZE, False) for t in ALLT}
REH   = {}
for t in ALLT:
    REH[t], n = sc.build_rehearsal_loader(SPLITS_DIR, t, fraction=REHEARSAL_FRACTION,
                                          reh_batch=REH_BATCH, image_size=IMAGE_SIZE,
                                          workers=WORKERS, seed=SEED, path_remap=REMAP)
    print(f"  rehearsal[{t}] = {n}")


# ===== CELL 4 — one arm = one (model, order) chain with a fixed-width head =====
SESSION_START = time.time()
results, matrices = [], {}

def run_arm(name, order_key):
    order = ORDERS[order_key]
    print("\n" + "=" * 74 + f"\n### {name} | {order_key}: {' -> '.join(order)}\n" + "=" * 74, flush=True)
    t0 = time.time()
    R = [[None] * len(order) for _ in range(len(order))]     # rows/cols indexed by POSITION in this order
    model = None
    for step, task in enumerate(order):
        # fixed 19-wide head throughout, so head size cannot confound the comparison
        student = sc.build_model(name, NUM_ALL, DEVICE, SEED,
                                 pretrained=(step == 0), feature_extract=True)
        if model is not None:
            student, _ = sc.transfer_backbone(model, student, name)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=LR)
        seen = order[:step + 1]
        if step == 0:
            student, _ = sc.train_naive(student, TRAIN[task], eval_loaders={task: VAL[task]},
                                        optimizer=opt, device=DEVICE, epochs=EPOCHS,
                                        model_name=f"{name}_{order_key}", logger=logger,
                                        experiment=f"{order_key}_base", select_best=True,
                                        select_loaders={task: VAL[task]},
                                        all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        else:
            student, _ = sc.train_rehearsal(student, new_task_loader=TRAIN[task],
                                            rehearsal_loaders=[REH[p] for p in order[:step]],
                                            eval_loaders={t: VAL[t] for t in seen}, optimizer=opt,
                                            device=DEVICE, epochs=EPOCHS,
                                            model_name=f"{name}_{order_key}", logger=logger,
                                            select_loaders=None, best_ckpt_path=None,
                                            all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        for i, t in enumerate(seen):
            R[i][step] = sc.evaluate_on_task(student, TEST[t], cs.task_idx(t),
                                             cs.all_classes, DEVICE)["accuracy"]
        del model; model = student
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"  after {task}: " + "  ".join(f"{order[i]}:{R[i][step]:.3f}" for i in range(step + 1)),
              flush=True)
    N = len(order)
    fin = [R[i][N - 1] for i in range(N)]
    bwt = [R[i][N - 1] - R[i][i] for i in range(N - 1)]
    fm  = [max(v for v in R[i][i:] if v is not None) - R[i][N - 1] for i in range(N - 1)]
    rec = {"model": name, "order": order_key, "sequence": " -> ".join(order),
           "ACC": float(np.mean(fin)), "BWT": float(np.mean(bwt)), "FM": float(np.mean(fm)),
           **{f"final_{t}": R[i][N - 1] for i, t in enumerate(order)},
           "minutes": round((time.time() - t0) / 60, 1)}
    results.append(rec); matrices.setdefault(name, {})[order_key] = {"order": order, "R": R}
    print(f"  => ACC {rec['ACC']:.4f}  BWT {rec['BWT']:.4f}  FM {rec['FM']:.4f} "
          f"({rec['minutes']} min)", flush=True)
    sc.save_checkpoint(model, f"{WORK}/models/{order_key}_final_{name}.pth",
                       cs.all_classes, seed=SEED, extra={"order": order})
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_taskorder.csv")
for name in MODELS_TO_RUN:
    for ok in ORDERS:
        try:
            run_arm(name, ok)
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}/{ok}: {e}"); traceback.print_exc()
        if (time.time() - SESSION_START) / 60 > SESSION_MAX_MINUTES:
            print("  [budget] session limit reached — rerun with the remaining arms."); break


# ===== CELL 5 — the comparison that tests the hypothesis =====
df = pd.DataFrame(results)
def _safe(s, f):
    try: f(); print("  wrote", s)
    except Exception as e: print(f"  [warn] {s}: {e}")
_safe("taskorder_results.csv", lambda: df.to_csv(f"{WORK}/csv/taskorder_results.csv", index=False))
_safe("taskorder_matrices.json", lambda: json.dump(matrices, open(f"{WORK}/csv/taskorder_matrices.json", "w"), indent=2))
_safe("results.xlsx", lambda: sc.build_results_xlsx({"taskorder": results}, f"{WORK}/xlsx/TASKORDER_results.xlsx"))

verdict = []
if not df.empty and {"original", "skin_first"} <= set(df.order):
    print("\n=== HYPOTHESIS TEST ===")
    print("prediction: moving the dissimilar (skin) task to the FRONT makes every remaining")
    print("transition within-modality, so total forgetting should INCREASE (lower ACC, more")
    print("negative BWT) relative to the original order.\n")
    for name in df.model.unique():
        a = df[(df.model == name) & (df.order == "original")]
        b = df[(df.model == name) & (df.order == "skin_first")]
        if a.empty or b.empty: continue
        dACC = float(b.ACC.iloc[0] - a.ACC.iloc[0]); dBWT = float(b.BWT.iloc[0] - a.BWT.iloc[0])
        supports = (dACC < -0.02) and (dBWT < 0)
        verdict.append({"model": name, "ACC_original": float(a.ACC.iloc[0]),
                        "ACC_skin_first": float(b.ACC.iloc[0]), "delta_ACC": dACC,
                        "delta_BWT": dBWT, "supports_hypothesis": bool(supports)})
        print(f"  {name:<11} ACC {a.ACC.iloc[0]:.4f} -> {b.ACC.iloc[0]:.4f}  (Δ {dACC:+.4f}) | "
              f"ΔBWT {dBWT:+.4f} | {'SUPPORTS' if supports else 'does not support'}")
    print("\n  Note the 0.03 run-to-run threshold from Section 2.13: a |ΔACC| below that is")
    print("  not resolvable and should be reported as 'no detectable effect of order'.")
    pd.DataFrame(verdict).to_csv(f"{WORK}/csv/taskorder_verdict.csv", index=False)

def fig():
    if df.empty: return
    fig, ax = plt.subplots(figsize=(2 + 1.6 * len(df.model.unique()), 4.4))
    models = list(df.model.unique()); x = np.arange(len(models)); w = 0.35
    for k, ok in enumerate(["original", "skin_first"]):
        v = [float(df[(df.model == m) & (df.order == ok)].ACC.iloc[0])
             if not df[(df.model == m) & (df.order == ok)].empty else np.nan for m in models]
        ax.bar(x + (k - 0.5) * w, v, w, label=ok.replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0, 1)
    ax.set_ylabel("average accuracy after the full sequence")
    ax.set_title("Does the order of tasks change total forgetting?"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/taskorder.png", dpi=300); plt.close(fig)
    print("  fig taskorder")
_safe("figure", fig)

zip_path = sc.bundle_outputs("TASKORDER", WORK, out_zip="/kaggle/working/TASKORDER_outputs.zip",
                             run_manifest={"experiment": "task_order", "orders": ORDERS,
                                           "models": MODELS_TO_RUN, "fixed_head": NUM_ALL,
                                           "seed": SEED, "epochs": EPOCHS})
print("\nBUNDLED ->", zip_path)
if not df.empty:
    print("\n=== TASK-ORDER ABLATION ===")
    print(df.round(4).to_string(index=False))
