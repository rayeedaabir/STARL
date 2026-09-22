# =============================================================================
# STARL v2 — EXEMPLAR-BUDGET ABLATION  (professor #5: "one instance per class")
# =============================================================================
# Our main rehearsal keeps 10% of every prior task. That's a big memory budget.
# This asks the memory-efficiency question: how FEW stored images per class can we
# keep and still avoid forgetting? We re-run the T2->T3->T4 continual chain with a
# fixed budget of m exemplars PER CLASS, for m in {1, 5, 20}, selected two ways:
#
#   * random  — pick m images of the class at random
#   * herding — iCaRL's rule: greedily pick the image that keeps the running mean of
#               chosen exemplars closest to the true class mean in feature space
#               (i.e. the m most "representative" images)
#
# m=1 IS LITERALLY THE PROFESSORS' "one instance per class" EXPERIMENT.
# Output: the memory-efficiency frontier (retention vs images stored per class).
#
# ATTACH: starl-code, starl-splits, all four image datasets, and the T1 output zip
# (needs expert_T1_APTOS_<model>.pth to start each chain). GPU required.
#
# COMPUTE: each config = a full T2->T4 chain. configs = |m| x |methods| = 6 per model.
# Start with ONE fast model to gauge timing, then add the second.


# ===== CELL 1 — imports =====
import sys, os, glob, json, time, copy
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# ===== CELL 2 — config =====
SEED, EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 15, 1e-4, 32, 224, 4
USE_AMP = True
MODELS_TO_RUN   = ["resnet18"]              # fast + a strong retainer; add "googlenet" next
BUDGETS         = [1, 5, 20]                # exemplars per class (m). m=1 = professors' ask
SELECTIONS      = ["herding", "random"]
REH_BATCH       = 4                         # replay images per training batch, per prior task
WORK = "/kaggle/working/EXEMPLAR"
for sub in ("csv", "xlsx", "docx", "figures", "models"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
CHAIN = TASKS[1:]                            # T2, T3, T4 (T1 expert is the starting point)
print("device:", DEVICE, "| chain:", CHAIN)


# ===== CELL 3 — path resolver + loaders =====
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff"); _INPUT_IDX = None
def _build_input_index(root="/kaggle/input"):
    idx = {}
    for r, _d, files in os.walk(root):
        for f in files:
            if f.lower().endswith(_IMG_EXT): idx.setdefault(f, os.path.join(r, f))
    return idx
def REMAP(path):
    if os.path.exists(path): return path
    global _INPUT_IDX
    if _INPUT_IDX is None:
        _INPUT_IDX = _build_input_index(); print(f"  indexed {len(_INPUT_IDX)} images", flush=True)
    return _INPUT_IDX.get(os.path.basename(path), path)

train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def rows_of(task, split): return sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split)
def make_loader(rows, tf, batch, shuffle):
    return sc._loader(rows, tf, batch, shuffle, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)

TRAIN_L = {t: make_loader(rows_of(t, "train"), train_tf, BATCH_SIZE, True)  for t in TASKS}
VAL_L   = {t: make_loader(rows_of(t, "val"),   eval_tf,  BATCH_SIZE, False) for t in TASKS}
TEST_L  = {t: make_loader(rows_of(t, "test"),  eval_tf,  BATCH_SIZE, False) for t in TASKS}
for t in TASKS:
    assert os.path.exists(REMAP(rows_of(t, "train")[0][0])), f"{t}: images missing"
print("  loaders ready")


# ===== CELL 4 — exemplar selection: random and herding (iCaRL) =====
def _feature_extractor(model, name):
    """Return a function images->(N,D) penultimate features, for herding.

    We hook the CLASSIFIER HEAD and capture its INPUT, which is the penultimate feature
    vector for every architecture in this project. Hooking the backbone's OUTPUT does not
    work for the timm transformers: TimmTransformerWrapper.forward calls
    `self.backbone.forward_features(x)` directly, which bypasses __call__, so a forward
    hook on the backbone never fires (swin_tiny / coatnet_0 would raise KeyError).
    sc._head_module() resolves the head uniformly: .head (transformers), .kan_head (KAN),
    .fc (resnet/googlenet), .classifier[6] (vgg/alexnet)."""
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

def select_exemplars(model, model_name, task, m, method, seed=SEED):
    """Return <= m*|classes| rows [(path,label)] from `task`'s TRAIN split."""
    rows = rows_of(task, "train")
    by_class = {}
    for p, l in rows: by_class.setdefault(l, []).append((p, l))
    rng = np.random.default_rng(seed)
    chosen = []
    if method == "random":
        for l, items in by_class.items():
            idx = rng.choice(len(items), size=min(m, len(items)), replace=False)
            chosen += [items[i] for i in idx]
        return chosen
    # ---- herding: needs features, so run the model over this task's train images ----
    model.eval()
    extract, handle = _feature_extractor(model, model_name)
    try:
        for l, items in by_class.items():
            if len(items) <= m:
                chosen += items; continue
            ld = make_loader(items, eval_tf, BATCH_SIZE, False)
            F = []
            for images, _lab in ld: F.append(extract(images).cpu())
            F = torch.cat(F).numpy()
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)   # iCaRL uses L2-normalised feats
            mu = F.mean(0)
            picked, running = [], np.zeros_like(mu)
            for k in range(min(m, len(items))):
                # pick the item making the running mean closest to the class mean
                cand = (mu - (running[None, :] + F) / (k + 1))
                d = np.linalg.norm(cand, axis=1)
                d[picked] = np.inf
                j = int(np.argmin(d))
                picked.append(j); running = running + F[j]
            chosen += [items[j] for j in picked]
    finally:
        handle.remove()
    return chosen


# ===== CELL 5 — the continual chain under a fixed exemplar budget =====
def load_expert(model_name, task):
    hits = glob.glob(f"/kaggle/input/**/expert_{task}_{model_name}.pth", recursive=True)
    if not hits: return None
    m = sc.build_model(model_name, cs.num_classes_through(task), DEVICE, SEED,
                       pretrained=False, feature_extract=True)
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m

# The ablation only varies the REPLAY BUFFER used in T2->T4. T1 is the base task with no
# prior tasks, so it has no buffer and is identical for all 6 configs — reusing the one
# existing T1 expert avoids retraining the same model 6x AND makes this a controlled
# experiment where the buffer is the only variable. If the T1 zip isn't attached, we can
# train that base model once here and reuse it for every config (set the flag below).
TRAIN_T1_IF_MISSING = True
_T1_CACHE = {}
def get_t1_expert(model_name):
    if model_name in _T1_CACHE: return _T1_CACHE[model_name]
    m = load_expert(model_name, TASKS[0])
    if m is None:
        assert TRAIN_T1_IF_MISSING, (
            f"No expert_{TASKS[0]}_{model_name}.pth found — attach the T1 output zip "
            f"or set TRAIN_T1_IF_MISSING=True to train the base model once here.")
        print(f"  [T1] no checkpoint found — training the {model_name} base model ONCE "
              f"(reused by all configs)...", flush=True)
        m = sc.build_model(model_name, cs.num_classes_through(TASKS[0]), DEVICE, SEED,
                           pretrained=True, feature_extract=True)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=LR)
        m, _ = sc.train_naive(m, TRAIN_L[TASKS[0]], eval_loaders={TASKS[0]: VAL_L[TASKS[0]]},
                              optimizer=opt, device=DEVICE, epochs=EPOCHS, model_name=model_name,
                              logger=epoch_logger, experiment="t1_base", select_best=True,
                              best_ckpt_path=f"{WORK}/models/expert_{TASKS[0]}_{model_name}.pth",
                              all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
    _T1_CACHE[model_name] = m
    return m

def cl_metrics_from_R(R):
    N = len(R)
    fin = [R[i][N-1] for i in range(N) if R[i][N-1] is not None]
    bwt = [R[i][N-1] - R[i][i] for i in range(N-1) if R[i][N-1] is not None and R[i][i] is not None]
    fm  = [max(v for v in (R[i][j] for j in range(i, N)) if v is not None) - R[i][N-1]
           for i in range(N-1) if R[i][N-1] is not None]
    return {"ACC": float(np.mean(fin)) if fin else float("nan"),
            "BWT": float(np.mean(bwt)) if bwt else float("nan"),
            "FM":  float(np.mean(fm))  if fm  else float("nan")}

epoch_logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_EXEMPLAR.csv")
result_rows, buffer_rows = [], []

def run_config(model_name, m, method):
    tag = f"{model_name}|m={m}|{method}"
    print("\n" + "=" * 74 + f"\n### {tag}\n" + "=" * 74, flush=True)
    t0 = time.time()
    expert = get_t1_expert(model_name)      # same T1 start for every config (controlled experiment)
    R = [[None] * len(TASKS) for _ in range(len(TASKS))]
    # T1 diagonal: the starting expert's own accuracy
    R[0][0] = sc.evaluate_on_task(expert, TEST_L[TASKS[0]], cs.task_idx(TASKS[0]), cs.all_classes, DEVICE)["accuracy"]
    buffers = {}          # task -> exemplar rows kept for replay
    model = expert
    for step, task in enumerate(CHAIN, start=1):
        prev = TASKS[step - 1]
        # select this budget's exemplars from the task we just finished, using the CURRENT model
        buffers[prev] = select_exemplars(model, model_name, prev, m, method)
        buffer_rows.append({"model": model_name, "m": m, "selection": method, "task": prev,
                            "n_exemplars": len(buffers[prev]),
                            "n_classes": len({l for _, l in buffers[prev]})})
        print(f"  buffer[{prev}] = {len(buffers[prev])} images "
              f"({len({l for _,l in buffers[prev]})} classes, m={m}, {method})", flush=True)
        # grow the head for the new task and carry the backbone over
        student = sc.build_model(model_name, cs.num_classes_through(task), DEVICE, SEED,
                                 pretrained=False, feature_extract=True)
        student, _ = sc.transfer_backbone(model, student, model_name)
        reh_loaders = [make_loader(buffers[t], train_tf, REH_BATCH, True) for t in buffers]
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=LR)
        seen = TASKS[:step + 1]
        student, _ = sc.train_rehearsal(
            student, new_task_loader=TRAIN_L[task], rehearsal_loaders=reh_loaders,
            eval_loaders={t: VAL_L[t] for t in seen}, optimizer=opt, device=DEVICE, epochs=EPOCHS,
            model_name=f"{model_name}_m{m}_{method}", logger=epoch_logger, select_loaders=None,
            best_ckpt_path=None, all_classes=cs.all_classes, seed=SEED, use_amp=USE_AMP)
        for i, t in enumerate(seen):
            R[i][step] = sc.evaluate_on_task(student, TEST_L[t], cs.task_idx(t), cs.all_classes, DEVICE)["accuracy"]
        del model; model = student
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    met = cl_metrics_from_R(R)
    total_stored = sum(len(v) for v in buffers.values())
    result_rows.append({"model": model_name, "m": m, "selection": method,
                        "images_stored_total": total_stored, **met,
                        "final_per_task": json.dumps({TASKS[i]: R[i][-1] for i in range(len(TASKS))}),
                        "minutes": round((time.time() - t0) / 60, 1)})
    print(f"  ✓ {tag}: ACC {met['ACC']:.4f} BWT {met['BWT']:.4f} FM {met['FM']:.4f} "
          f"| stored {total_stored} images | {(time.time()-t0)/60:.1f} min", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

for mn in MODELS_TO_RUN:
    for m in BUDGETS:
        for method in SELECTIONS:
            try:
                run_config(mn, m, method)
            except Exception as e:
                import traceback; print(f"  [ERROR] {mn} m={m} {method}: {e}"); traceback.print_exc()


# ===== CELL 6 — frontier figure, tables, bundle =====
res_df = pd.DataFrame(result_rows)
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")
_safe("exemplar_results.csv", lambda: res_df.to_csv(f"{WORK}/csv/exemplar_results.csv", index=False))
_safe("exemplar_buffers.csv", lambda: pd.DataFrame(buffer_rows).to_csv(f"{WORK}/csv/exemplar_buffers.csv", index=False))
_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"exemplar_results": result_rows, "buffers": buffer_rows}, f"{WORK}/xlsx/EXEMPLAR_results.xlsx"))

def fig_frontier():
    if res_df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, met, ttl in zip(axes, ["ACC", "FM"], ["Average accuracy ↑", "Forgetting ↓"]):
        for (mn, sel), g in res_df.groupby(["model", "selection"]):
            g = g.sort_values("m")
            ax.plot(g.m, g[met], marker="o", label=f"{mn}·{sel}")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlabel("exemplars stored per class (m)"); ax.set_title(ttl)
    axes[0].set_ylabel("after full sequence"); axes[0].legend(fontsize=8)
    fig.suptitle("Memory-efficiency frontier: how few stored images still prevent forgetting?", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/exemplar_frontier.png", dpi=300); plt.close(fig)
    print("  fig exemplar_frontier")
_safe("figures", fig_frontier)
_safe("report.docx", lambda: sc.build_report_docx("EXEMPLAR", result_rows, f"{WORK}/docx/EXEMPLAR_report.docx",
                                                  figures_dir=f"{WORK}/figures",
                                                  title="STARL v2 — Exemplar-budget ablation (m=1 included)"))

manifest = {"experiment": "exemplar_budget", "models": MODELS_TO_RUN, "budgets": BUDGETS,
            "selections": SELECTIONS, "epochs": EPOCHS, "chain": CHAIN, "reh_batch": REH_BATCH}
zip_path = sc.bundle_outputs("EXEMPLAR", WORK, out_zip="/kaggle/working/EXEMPLAR_outputs.zip", run_manifest=manifest)
print("\nBUNDLED ->", zip_path)
if not res_df.empty:
    print("\n=== EXEMPLAR-BUDGET ABLATION ===")
    print(res_df[["model", "m", "selection", "images_stored_total", "ACC", "BWT", "FM", "minutes"]]
          .round(4).to_string(index=False))
