# =============================================================================
# STARL v2 — CKA ACTIVATION-PATH STUDY  (professor #11)
# =============================================================================
# Tests the professors' "seen and unseen images of the same class follow the same
# activation paths" hypothesis, and gives a mechanism for WHY rehearsal retains.
# Two complementary, rigorously-defined measures per layer:
#
#   (A) SEEN-vs-UNSEEN consistency  — per class, cosine( mean train-image feature,
#       mean test-image feature ). High = the model routes seen & unseen images of a
#       class through the same features. (Valid for unpaired train/test sets.)
#
#   (B) REPRESENTATION DRIFT (standard linear + RBF CKA, PAIRED same inputs) — for a
#       prior task's test images, CKA between the model that first learned that task and
#       the FINAL model. High = the task's activation paths were PRESERVED as later tasks
#       were learned. This is the retention mechanism: rehearsal keeps CKA high.
#
# Only rehearsal checkpoints were saved during the runs, so (B) is shown for rehearsal
# by default. Set RUN_NAIVE_COMPARISON=True to also train quick naive models here (load
# prev expert -> naive on that task) for the naive-vs-rehearsal contrast the professors
# want (naive should drift -> low CKA).
#
# ATTACH: starl-code, starl-splits, the image datasets, and the task output zips that
# hold the rehearsal experts (expert_T1_APTOS_*, ... expert_T4_HAM_*). GPU recommended.


# ===== CELL 1 — imports =====
import sys, os, json, glob, time
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())


# ===== CELL 2 — config =====
SEED, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 32, 224, 4
N_CKA = 512                       # images/task for CKA; larger = lower finite-sample bias on wide layers
MODELS_TO_RUN = ["resnet50"]      # ResNet-family hooks below; add resnet18/101/152 freely
# RESNET layer taps (clean, interpretable "activation path"):
RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
RUN_NAIVE_COMPARISON = False      # True = also train quick naive models for the contrast
EPOCHS_NAIVE = 15
LR = 1e-4

WORK = "/kaggle/working/CKA"
for sub in ("csv", "figures", "docx", "xlsx", "models"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print("device:", DEVICE)


# ===== CELL 3 — class space, path resolver, per-task loaders =====
cs    = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
NUM   = len(cs.all_classes)

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
def loader(task, split, tf=None):
    return sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split),
                      tf or eval_tf, BATCH_SIZE, False, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
for t in TASKS:
    assert os.path.exists(REMAP(sc.read_manifest(sc.manifest_path(SPLITS_DIR, t))[0][0])), f"{t}: images missing"
print("  images resolve OK | tasks:", TASKS)


# ===== CELL 4 — CKA (linear + RBF) and layer-feature extraction via hooks =====
def _center(K):
    n = K.shape[0]; H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H
def _hsic(K, L):
    return float(np.sum(_center(K) * _center(L)))
def linear_cka(X, Y):
    K, L = X @ X.T, Y @ Y.T
    d = np.sqrt(_hsic(K, K) * _hsic(L, L))
    return _hsic(K, L) / d if d > 0 else float("nan")
def _rbf_gram(X):
    sq = np.sum(X ** 2, 1); D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
    return np.exp(-D / (med + 1e-12))
def rbf_cka(X, Y):
    K, L = _rbf_gram(X), _rbf_gram(Y)
    d = np.sqrt(_hsic(K, K) * _hsic(L, L))
    return _hsic(K, L) / d if d > 0 else float("nan")

def layer_features(model, ld, layer_names, n_max=N_CKA):
    """Return {layer: (N,C) pooled features} and the label vector, over up to n_max images."""
    store, feats, ys = {}, {ln: [] for ln in layer_names}, []
    mods = dict(model.named_modules())
    def mk(ln):
        def hook(_m, _i, o):
            x = o
            if x.dim() == 4: x = x.mean(dim=[2, 3])
            elif x.dim() == 3: x = x.mean(dim=1)
            store[ln] = x.detach().float().cpu()
        return hook
    handles = [mods[ln].register_forward_hook(mk(ln)) for ln in layer_names]
    model.eval(); n = 0
    with torch.no_grad():
        for imgs, labels in ld:
            model(imgs.to(DEVICE))
            for ln in layer_names: feats[ln].append(store[ln])
            ys.append(labels); n += imgs.size(0)
            if n >= n_max: break
    for h in handles: h.remove()
    out = {ln: torch.cat(feats[ln])[:n_max].numpy() for ln in layer_names}
    return out, torch.cat(ys)[:n_max].numpy()


# ===== CELL 5 — locate the rehearsal experts for each (model, task) =====
def find_expert(model, task):
    hits = glob.glob(f"/kaggle/input/**/expert_{task}_{model}.pth", recursive=True)
    return hits[0] if hits else None

def load_model_at(model_name, task):
    """The rehearsal model as it stood after `task` = build at that task's head size + load."""
    path = find_expert(model_name, task)
    if path is None: return None
    num = cs.num_classes_through(task)
    m = sc.build_model(model_name, num, DEVICE, SEED, pretrained=False, feature_extract=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m


# ===== CELL 6 — (A) seen-vs-unseen consistency + (B) representation drift =====
def _class_centroids(feat, y, classes):
    return {c: feat[y == c].mean(0) for c in classes if np.any(y == c)}
def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

consistency_rows, drift_rows = [], []

def analyse(model_name, tag="rehearsal", model_provider=load_model_at):
    final_model = model_provider(model_name, TASKS[-1])
    if final_model is None:
        print(f"  [skip] {model_name}: no {TASKS[-1]} expert found"); return
    # (A) seen (train) vs unseen (test) per layer, per task, on the FINAL model
    for t in TASKS:
        ftr, ytr = layer_features(final_model, loader(t, "train", train_tf), RESNET_LAYERS)
        fte, yte = layer_features(final_model, loader(t, "test"),  RESNET_LAYERS)
        for ln in RESNET_LAYERS:
            ctr = _class_centroids(ftr[ln], ytr, cs.task_idx(t))
            cte = _class_centroids(fte[ln], yte, cs.task_idx(t))
            sims = [_cos(ctr[c], cte[c]) for c in ctr if c in cte]
            consistency_rows.append({"model": model_name, "strategy": tag, "task": t, "layer": ln,
                                     "seen_unseen_cosine": float(np.mean(sims)) if sims else float("nan")})
    # (B) drift: for each prior task, CKA(model-that-learned-it vs final model) on its TEST images
    for t in TASKS[:-1]:
        early = model_provider(model_name, t)
        if early is None: continue
        ld = loader(t, "test")
        fe, _ = layer_features(early, ld, RESNET_LAYERS)
        ff, _ = layer_features(final_model, ld, RESNET_LAYERS)
        for ln in RESNET_LAYERS:
            drift_rows.append({"model": model_name, "strategy": tag, "task": t, "layer": ln,
                               "cka_linear": linear_cka(fe[ln], ff[ln]),
                               "cka_rbf": rbf_cka(fe[ln], ff[ln])})
        del early
    del final_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"  ✓ CKA analysis done for {model_name} ({tag})", flush=True)

for mname in MODELS_TO_RUN:
    try: analyse(mname, "rehearsal")
    except Exception as e:
        import traceback; print(f"  [ERROR] {mname}: {e}"); traceback.print_exc()


# ===== CELL 7 — OPTIONAL naive-vs-rehearsal contrast (trains quick naive models here) =====
if RUN_NAIVE_COMPARISON:
    import starl_baselines as sb  # noqa
    naive_cache = {}   # (model, task) -> trained naive model (from prev rehearsal expert)
    def naive_model_at(model_name, task):
        if (model_name, task) in naive_cache: return naive_cache[(model_name, task)]
        ti = TASKS.index(task)
        prev = load_model_at(model_name, TASKS[ti - 1]) if ti > 0 else None
        num = cs.num_classes_through(task)
        m = sc.build_model(model_name, num, DEVICE, SEED, pretrained=(ti == 0), feature_extract=True)
        if prev is not None: m, _ = sc.transfer_backbone(prev, m, model_name)
        tr = sc._loader(sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), "train"),
                        train_tf, BATCH_SIZE, True, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=LR)
        m, _ = sc.train_naive(m, tr, eval_loaders={task: loader(task, "val")}, optimizer=opt, device=DEVICE,
                              epochs=EPOCHS_NAIVE, model_name=f"{model_name}_naive", experiment="naive",
                              select_best=True, all_classes=cs.all_classes, seed=SEED)
        naive_cache[(model_name, task)] = m
        return m
    for mname in MODELS_TO_RUN:
        try: analyse(mname, "naive", model_provider=naive_model_at)
        except Exception as e:
            import traceback; print(f"  [ERROR] naive {mname}: {e}"); traceback.print_exc()


# ===== CELL 8 — figures, tables, bundle =====
import pandas as pd
cons_df, drift_df = pd.DataFrame(consistency_rows), pd.DataFrame(drift_rows)

def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")
_safe("cka_seen_unseen.csv", lambda: cons_df.to_csv(f"{WORK}/csv/cka_seen_unseen.csv", index=False))
_safe("cka_drift.csv",       lambda: drift_df.to_csv(f"{WORK}/csv/cka_drift.csv", index=False))

def fig_drift():
    if drift_df.empty: return
    for mname in drift_df.model.unique():
        d = drift_df[drift_df.model == mname]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (strat, t), g in d.groupby(["strategy", "task"]):
            g = g.set_index("layer").reindex(RESNET_LAYERS)
            ax.plot(RESNET_LAYERS, g.cka_linear, marker="o", label=f"{strat}·{t}")
        ax.set_ylim(0, 1.02); ax.set_ylabel("linear CKA (early expert vs final)")
        ax.set_title(f"{mname}: activation-path preservation across the sequence")
        ax.set_xlabel("ResNet stage"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(f"{WORK}/figures/cka_drift_{mname}.png", dpi=300); plt.close(fig)
        print(f"  fig cka_drift_{mname}")
def fig_consistency():
    if cons_df.empty: return
    piv = cons_df.pivot_table(index="layer", columns="task", values="seen_unseen_cosine").reindex(RESNET_LAYERS)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                        color="white" if v < 0.6 else "black")
    ax.set_title("Seen-vs-unseen feature consistency (cosine of class centroids)")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(f"{WORK}/figures/cka_seen_unseen.png", dpi=300); plt.close(fig); print("  fig cka_seen_unseen")
for fn in (fig_drift, fig_consistency):
    try: fn()
    except Exception as e: print(f"  [warn] {fn.__name__}: {e}")

_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"seen_unseen_consistency": consistency_rows, "representation_drift": drift_rows}, f"{WORK}/xlsx/CKA_results.xlsx"))
manifest = {"experiment": "cka", "models": MODELS_TO_RUN, "layers": RESNET_LAYERS, "n_cka": N_CKA,
            "naive_comparison": RUN_NAIVE_COMPARISON, "tasks": TASKS}
zip_path = sc.bundle_outputs("CKA", WORK, out_zip="/kaggle/working/CKA_outputs.zip", run_manifest=manifest)
print("\nBUNDLED ->", zip_path)
if not drift_df.empty:
    print("\n=== representation drift (linear CKA: early expert vs final, higher = paths preserved) ===")
    print(drift_df.round(3).to_string(index=False))
if not cons_df.empty:
    print("\n=== seen-vs-unseen consistency (cosine) ===")
    print(cons_df.round(3).to_string(index=False))
