# =============================================================================
# STARL v2 — KAN CHAIN RUNNER  (own account, resumable, one templated file)
# =============================================================================
# KAN is slow (~24 min/epoch) and its custom ops don't autocast, so this runner
# is SEPARATE from the 11-model runners: AMP is OFF and it check-points every
# epoch so a task can span several Kaggle sessions.
#
# It runs the WHOLE KAN chain, one TASK per notebook, same as the others:
#   T1_APTOS  -> base training (+ frozen-probe)         -> expert_T1_APTOS_kan.pth
#   T2_ODIR   -> naive/rehearsal/lwf/ewc/probe/indep    -> expert_T2_ODIR_kan.pth
#   T3_LAG    -> (same six)                             -> expert_T3_LAG_kan.pth
#   T4_HAM    -> (same six)                             -> expert_T4_HAM_kan.pth
#
# ------------------------------ RESUME MODEL ---------------------------------
# Kaggle wipes /kaggle/working between sessions and kills a run at ~12h, so:
#   * every epoch it writes resume/resume_state.pt and RE-BUNDLES the output zip
#   * after SESSION_MAX_MINUTES it stops CLEANLY (so the commit saves its output)
# To continue: download {TASK}_kan_outputs.zip, upload it as a Kaggle dataset,
# attach it, set RESUME=True in CELL 2, and Save & Run All again. It picks up at
# the exact stage+epoch it left off (model, optimizer, best-so-far, RNG, results).
#
# ATTACH (Add Input):
#   starl-code, starl-splits, the image datasets for THIS task (T1:APTOS;
#   T2:+ODIR; T3:+LAG; T4:+HAM), the PREVIOUS task's expert zip (T2+ only),
#   and — when RESUME=True — the uploaded {TASK}_kan_outputs.zip resume dataset.


# ===== CELL 1 — imports & attach the code library =====
import sys, os, json, time, glob, copy, shutil, zipfile
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
import starl_baselines as sb
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
# pykan + python-docx aren't preinstalled on Kaggle — install quietly (needs internet ON)
for pkg, mod in (("pykan", "kan"), ("python-docx", "docx")):
    try:
        __import__(mod)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])


# ===== CELL 2 — config =====
# ---- move along the chain by changing TASK (+ swap the attached expert zip) --
TASK       = "T1_APTOS"      # T1_APTOS -> T2_ODIR -> T3_LAG -> T4_HAM
RESUME     = False           # True to continue an unfinished TASK from its uploaded zip
RESUME_DIR = None            # None = auto-find resume_state.pt under /kaggle/input
EXPERT_DIR = None            # None = auto-find expert_<prev>_kan.pth (T2+ only)
# ----------------------------------------------------------------------------
SESSION_MAX_MINUTES = 660    # stop cleanly before Kaggle's ~12h cap so output saves
SEED       = 42
EPOCHS     = 15              # keep = the other models for a matched comparison
LR         = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = 224
WORKERS    = 4
USE_AMP    = False           # forced OFF for KAN (do not enable)

# strategies (CL tasks). Base task T1 always runs "base"; frozen-probe honoured below.
RUN_NAIVE, RUN_REHEARSAL, RUN_LWF, RUN_EWC, RUN_FROZEN_PROBE, RUN_INDEPENDENT = (
    True, True, True, True, True, True)
REHEARSAL_FRACTION, REH_BATCH = 0.10, 4
LWF_T, LWF_LAM = 2.0, 1.0
EWC_LAM = 1000.0
EWC_FISHER_BATCHES = 50       # reduced (KAN is slow); Fisher is an estimate anyway
PLOT_CM_STRATEGIES = {"naive", "rehearsal"}

MODEL = "kan"
WORK = f"/kaggle/working/{TASK}_kan"
CACHE_DIR = "/kaggle/working/img_cache_224"
DEVICE = sc.get_device()
sc.set_seed(SEED)
print("device:", DEVICE, "| task:", TASK, "| RESUME:", RESUME, "| output ->", WORK)


# ===== CELL 3 — class space, priors, path resolver, current-task data =====
cs       = sc.load_class_space(SPLITS_DIR)
NUM      = cs.num_classes_through(TASK)
TASK_IDX = cs.task_idx(TASK)
PRIOR    = cs.prior_tasks(TASK)              # [] for the base task T1
PREV     = PRIOR[-1] if PRIOR else None
SEQ      = PRIOR + [TASK]
IS_BASE  = (len(PRIOR) == 0)
print(f"{TASK}: {NUM} classes | prior: {PRIOR} | {'BASE task' if IS_BASE else 'prev expert: '+PREV}")

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
    if os.path.exists(path):
        return path
    global _INPUT_IDX
    if _INPUT_IDX is None:
        print("  building /kaggle/input image index (first path miss)...", flush=True)
        _INPUT_IDX = _build_input_index()
        print(f"  indexed {len(_INPUT_IDX)} images", flush=True)
    return _INPUT_IDX.get(os.path.basename(path), path)

for t in SEQ:
    _p0 = sc.read_manifest(sc.manifest_path(SPLITS_DIR, t))[0][0]
    assert os.path.exists(REMAP(_p0)), (
        f"{t}: image not found ({os.path.basename(_p0)}). Attach the SAME dataset used in Phase 1.")
print("  all task sample images resolve OK")

tr_loader, va_loader, te_loader = sc.build_task_loaders(
    SPLITS_DIR, TASK, BATCH_SIZE, IMAGE_SIZE, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)
print(f"  {TASK}: train {len(tr_loader.dataset)} | val {len(va_loader.dataset)} | test {len(te_loader.dataset)}")

if not IS_BASE:
    if EXPERT_DIR is None:
        _h = glob.glob(f"/kaggle/input/**/expert_{PREV}_kan.pth", recursive=True)
        EXPERT_DIR = os.path.dirname(_h[0]) if _h else None
    assert EXPERT_DIR and os.path.isdir(EXPERT_DIR), (
        f"Need expert_{PREV}_kan.pth under /kaggle/input — attach {PREV}'s KAN output zip.")
    print("  expert dir:", EXPERT_DIR)


# ===== CELL 4 — prior-task loaders (CL tasks only) + eval dicts =====
train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def _mkloader(task, split, tf, batch, shuffle):
    rows = sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split)
    return sc._loader(rows, tf, batch, shuffle, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)

ALL_VAL  = {TASK: va_loader}
ALL_TEST = {TASK: te_loader}
rehearsal, prior_train = {}, {}
for p in PRIOR:
    ALL_VAL[p]  = _mkloader(p, "val",  eval_tf, BATCH_SIZE, False)
    ALL_TEST[p] = _mkloader(p, "test", eval_tf, BATCH_SIZE, False)
    rehearsal[p], _n = sc.build_rehearsal_loader(
        SPLITS_DIR, p, fraction=REHEARSAL_FRACTION, reh_batch=REH_BATCH,
        image_size=IMAGE_SIZE, workers=WORKERS, seed=SEED, path_remap=REMAP)
    if RUN_EWC:
        prior_train[p] = _mkloader(p, "train", eval_tf, BATCH_SIZE, True)
    print(f"  prior {p}: val {len(ALL_VAL[p].dataset)} | test {len(ALL_TEST[p].dataset)} | rehearsal {_n}")


# ===== CELL 5 — plotting + confusion helper =====
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


# ===== CELL 6 — resume machinery + the KAN training/finish functions =====
RUN_MANIFEST = {"task": TASK, "model": "kan", "prev_task": PREV, "prior_tasks": PRIOR, "seq": SEQ,
                "seed": SEED, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "use_amp": USE_AMP,
                "rehearsal_fraction": REHEARSAL_FRACTION, "reh_batch": REH_BATCH,
                "lwf": {"T": LWF_T, "lam": LWF_LAM}, "ewc_lam": EWC_LAM,
                "ewc_fisher_batches": EWC_FISHER_BATCHES}
RESUME_ZIP = f"/kaggle/working/{TASK}_kan_outputs.zip"

def bundle():
    return sc.bundle_outputs(TASK, WORK, out_zip=RESUME_ZIP, run_manifest=RUN_MANIFEST)

def _snapshot_rng():
    return {"torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "np": np.random.get_state()}
def _restore_rng(r):
    """Best-effort RNG restore. This is a reproducibility nicety, so it must NEVER be
    able to block a multi-hour resume — on any problem, warn and carry on."""
    if not r:
        return
    try:
        torch.set_rng_state(r["torch"].cpu().to(torch.uint8))          # must be a CPU ByteTensor
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in r["cuda"]])
        np.random.set_state(r["np"])
    except Exception as e:
        print(f"  [warn] RNG state not restored ({e}) — continuing, results unaffected.", flush=True)

def save_resume(state, rezip=True):
    os.makedirs(f"{WORK}/resume", exist_ok=True)
    tmp = f"{WORK}/resume/resume_state.pt.tmp"
    torch.save(state, tmp); os.replace(tmp, f"{WORK}/resume/resume_state.pt")   # atomic
    if rezip:
        bundle()

def _load_state(pt):
    # weights_only=False: this is OUR file and it holds numpy RNG state / plain dicts,
    # which torch>=2.6's default weights_only=True unpickler would reject.
    # map_location="cpu": the saved RNG states are CPU ByteTensors and torch.set_rng_state
    # REFUSES a CUDA tensor, so we must not relocate them. Loading the model/optimizer
    # state from CPU is fine: load_state_dict copies into the (already-GPU) module, and
    # Optimizer.load_state_dict casts its state to each param's device.
    return torch.load(pt, map_location="cpu", weights_only=False)

def restore_work_from_resume():
    """Find the resume file FOR THIS TASK and copy its bundle back into WORK.
    Every finished task's zip also contains a resume_state.pt, so we must match on
    state['task'] == TASK — not just grab the first one found (that was the T2-vs-T3 bug)."""
    base = RESUME_DIR or "/kaggle/input"
    # (a) extracted datasets: read each resume_state.pt and keep the ones for THIS task
    found = {}
    matches = []
    for pt in glob.glob(f"{base}/**/resume_state.pt", recursive=True):
        try:
            st = _load_state(pt)
        except Exception as e:
            print(f"  [warn] could not read {pt}: {e}"); continue
        found.setdefault(st.get("task"), 0); found[st.get("task")] += 1
        if st.get("task") == TASK:
            matches.append((os.path.dirname(os.path.dirname(pt)), st))   # (bundle_root, state)
    if matches:
        matches.sort(key=lambda rs: rs[1].get("task_done", False))       # prefer an UNFINISHED one
        root, st = matches[0]
        for item in os.listdir(root):
            s = os.path.join(root, item); d = os.path.join(WORK, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                os.makedirs(WORK, exist_ok=True); shutil.copy2(s, d)
        return _load_state(f"{WORK}/resume/resume_state.pt")
    # (b) fallback: the dataset kept the zip un-extracted — the zip name carries the task
    zips = glob.glob(f"{base}/**/{TASK}_kan_outputs.zip", recursive=True)
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(WORK)
        return _load_state(f"{WORK}/resume/resume_state.pt")
    raise FileNotFoundError(
        f"RESUME=True for {TASK}, but no resume_state.pt with task='{TASK}' was found under {base}.\n"
        f"  Resume files present are for tasks: {dict(found) or 'none'}.\n"
        f"  Attach THIS task's own output zip (the {TASK}_kan_outputs.zip you last downloaded), "
        f"not just the previous task's expert zip.")

def init_state():
    return {"task": TASK, "stage_idx": 0, "epoch": 0, "strategy": None,
            "latest_model": None, "latest_optim": None, "best_avg": -1.0, "best_state": None, "history": [],
            "expert_done": False, "expert_acc": {}, "fisher_ready": False,
            "final_rows": [], "perclass_rows": [], "clmetrics_rows": [], "cl_matrices": {},
            "after_by_strategy": {}, "completed": [], "task_done": False, "rng": None}

# ---- model construction (session_expert is set in the driver for CL tasks) ----
def _fresh_student(frozen_probe=False):
    m = sc.build_model(MODEL, NUM, DEVICE, SEED, pretrained=False, feature_extract=True, frozen_probe=frozen_probe)
    m, _ = sc.transfer_backbone(session_expert, m, MODEL)
    return m

def build_stage_model(strategy, resume_model_state=None):
    if strategy == "frozen_probe":
        model = (_fresh_student(frozen_probe=True) if not IS_BASE else
                 sc.build_model(MODEL, NUM, DEVICE, SEED, pretrained=True, feature_extract=True, frozen_probe=True))
    elif strategy == "independent":
        model = sc.build_model(MODEL, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
    elif strategy in ("naive", "rehearsal", "lwf", "ewc"):
        model = _fresh_student(frozen_probe=False)                 # CL: expert backbone transferred
    else:  # "base" (T1 only): fresh ImageNet backbone, fine-tuned
        model = sc.build_model(MODEL, NUM, DEVICE, SEED, pretrained=True, feature_extract=True)
    if resume_model_state is not None:
        model.load_state_dict(resume_model_state)
    return model

def load_expert():
    ckpt = torch.load(os.path.join(EXPERT_DIR, f"expert_{PREV}_{MODEL}.pth"),
                      map_location=DEVICE, weights_only=False)
    ex = sc.build_model(MODEL, cs.num_classes_through(PREV), DEVICE, SEED, pretrained=False, feature_extract=True)
    ex.load_state_dict(ckpt["model_state_dict"]); ex.eval()
    return ex

def stage_plan():
    if IS_BASE:
        return ["base"] + (["frozen_probe"] if RUN_FROZEN_PROBE else [])
    order = [("naive", RUN_NAIVE), ("rehearsal", RUN_REHEARSAL), ("lwf", RUN_LWF),
             ("ewc", RUN_EWC), ("frozen_probe", RUN_FROZEN_PROBE), ("independent", RUN_INDEPENDENT)]
    return [s for s, on in order if on]

def stage_config(strategy, fisher_list=None, star_list=None):
    cfg = {"eval": ALL_VAL, "select": None, "best_ckpt": None, "teacher": None,
           "n_old": None, "fisher": None, "star": None, "rehearsal": None}
    if strategy in ("base", "naive", "frozen_probe", "independent"):
        cfg["select"] = {TASK: va_loader}
    if strategy in ("base", "rehearsal"):
        cfg["best_ckpt"] = f"{WORK}/models/expert_{TASK}_{MODEL}.pth"   # <- the NEXT task's expert
    if strategy == "rehearsal":
        cfg["rehearsal"] = [rehearsal[p] for p in PRIOR]
    if strategy == "lwf":
        cfg["teacher"] = session_expert; cfg["n_old"] = cs.num_classes_through(PREV)
    if strategy == "ewc":
        cfg["fisher"] = fisher_list; cfg["star"] = star_list
    return cfg

class _Budget(Exception):
    pass

def run_stage(strategy, model, optimizer, start_epoch, cfg, state):
    """One strategy's epoch loop (AMP-free), checkpointing every epoch. Resumes from
    start_epoch (>0 continues an interrupted stage). Raises _Budget when the session
    clock passes SESSION_MAX_MINUTES so the commit can save its output cleanly."""
    criterion = nn.CrossEntropyLoss()
    if start_epoch > 0:
        best_avg, best_state, history = state["best_avg"], state["best_state"], state["history"]
    else:
        best_avg, best_state, history = -1.0, None, []
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time(); model.train(); run = correct = total = 0.0
        if strategy == "rehearsal":
            iters = [iter(rl) for rl in cfg["rehearsal"]]
            for s_img, s_lbl in tr_loader:
                imgs, lbls = [s_img], [s_lbl]
                for i, rl in enumerate(cfg["rehearsal"]):
                    try:
                        r_i, r_l = next(iters[i])
                    except StopIteration:
                        iters[i] = iter(rl); r_i, r_l = next(iters[i])
                    imgs.append(r_i); lbls.append(r_l)
                images = torch.cat(imgs).to(DEVICE); labels = torch.cat(lbls).to(DEVICE)
                optimizer.zero_grad()
                out = model(images); loss = criterion(out, labels)
                loss.backward(); optimizer.step()
                run += loss.item() * images.size(0)
                _, pr = torch.max(out, 1); correct += (pr == labels).sum().item(); total += images.size(0)
        else:
            for images, labels in tr_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(images); loss = criterion(out, labels)
                if strategy == "lwf":
                    with torch.no_grad():
                        t_out = cfg["teacher"](images)[:, :cfg["n_old"]]
                    s_old = F.log_softmax(out[:, :cfg["n_old"]] / LWF_T, dim=1)
                    t_sof = F.softmax(t_out / LWF_T, dim=1)
                    loss = loss + LWF_LAM * F.kl_div(s_old, t_sof, reduction="batchmean") * (LWF_T * LWF_T)
                elif strategy == "ewc":
                    loss = loss + (EWC_LAM / 2.0) * sb.ewc_penalty(model, cfg["fisher"], cfg["star"])
                loss.backward(); optimizer.step()
                run += loss.item() * images.size(0)
                _, pr = torch.max(out, 1); correct += (pr == labels).sum().item(); total += images.size(0)
        tr_loss, tr_acc = run / total, correct / total
        accs = {n: sc.quick_accuracy(model, ld, DEVICE) for n, ld in cfg["eval"].items()}
        row = {"model": MODEL, "experiment": strategy, "epoch": epoch, "time_s": round(time.time() - t0, 2),
               "train_loss": tr_loss, "train_acc": tr_acc}
        row.update({f"{n}_acc": accs[n] for n in cfg["eval"]})
        epoch_logger.log_epoch(row); history.append(row)
        print(f"  [{MODEL}/{strategy}] ep {epoch}/{EPOCHS} {row['time_s']:.0f}s loss {tr_loss:.4f} "
              f"acc {tr_acc:.4f} | " + " ".join(f"{n}:{a:.3f}" for n, a in accs.items()), flush=True)
        sel = ({n: sc.quick_accuracy(model, ld, DEVICE) for n, ld in cfg["select"].items()}
               if cfg["select"] else accs)
        avg = float(np.mean(list(sel.values())))
        if avg > best_avg:
            best_avg, best_state = avg, copy.deepcopy(model.state_dict())
            if cfg["best_ckpt"]:
                sc.save_checkpoint(model, cfg["best_ckpt"], cs.all_classes, seed=SEED,
                                   extra={"epoch": epoch, "avg_acc": avg})
        # ---- checkpoint the resume state after every epoch ----
        state.update({"strategy": strategy, "epoch": epoch, "best_avg": best_avg, "best_state": best_state,
                      "history": history, "latest_model": model.state_dict(),
                      "latest_optim": optimizer.state_dict(), "rng": _snapshot_rng()})
        save_resume(state)
        if (time.time() - SESSION_START) / 60.0 > SESSION_MAX_MINUTES:
            print(f"  [budget] {SESSION_MAX_MINUTES} min reached after {strategy} ep {epoch}; resume saved.", flush=True)
            raise _Budget()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ---- evaluation / recording (all accumulators live in `state` so resume restores them) ----
def _eval(model, task):
    return sc.evaluate_on_task(model, ALL_TEST[task], cs.task_idx(task), cs.all_classes, DEVICE)

def _record(state, experiment, eval_task, m, minutes=None):
    state["final_rows"].append({"model": MODEL, "experiment": experiment, "eval_task": eval_task,
                                "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                                "f1_macro": m["f1_macro"], "precision_weighted": m["precision_weighted"],
                                "recall_weighted": m["recall_weighted"],
                                "roc_auc": m["roc_auc_ovr_weighted"], "train_minutes": minutes})
    for cls, v in m["per_class"].items():
        state["perclass_rows"].append({"model": MODEL, "experiment": experiment, "eval_task": eval_task,
                                       "class": cls, **v})

def _stage_minutes(state):
    return round(sum(r["time_s"] for r in state["history"]) / 60.0, 2)

def finish_cl_stage(state, strategy, model):
    minutes = _stage_minutes(state); after = {}; m_cur = None
    for t in SEQ:
        m = _eval(model, t); after[t] = m["accuracy"]
        _record(state, strategy, t, m, minutes if t == TASK else None)
        if t == TASK:
            m_cur = m
    state["after_by_strategy"][strategy] = after
    if strategy in PLOT_CM_STRATEGIES and m_cur is not None:
        plot_confusion(m_cur["confusion_matrix"], m_cur["class_names"], f"kan — {TASK} {strategy} (test)",
                       f"{WORK}/figures/cm_{TASK}_{MODEL}_{strategy}.png")
    N = len(SEQ); R = [[None] * N for _ in range(N)]
    for i, ti in enumerate(SEQ):
        R[i][N - 1] = after[ti]
        if ti in state["expert_acc"]:
            R[i][N - 2] = state["expert_acc"][ti]
    clm = sc.cl_metrics(R)
    state["cl_matrices"].setdefault(MODEL, {})[strategy] = R
    state["clmetrics_rows"].append({"model": MODEL, "experiment": strategy,
                                    "average_accuracy": clm["average_accuracy"],
                                    "backward_transfer": clm["backward_transfer"],
                                    "forgetting_measure": clm["forgetting_measure"]})
    print(f"  [{strategy}] ACC {clm['average_accuracy']:.4f} BWT {clm['backward_transfer']:.4f} "
          f"FM {clm['forgetting_measure']:.4f}", flush=True)

def finish_base(state, model):
    m = _eval(model, TASK); _record(state, "base", TASK, m, _stage_minutes(state))
    plot_confusion(m["confusion_matrix"], m["class_names"], f"kan — {TASK} base (test)",
                   f"{WORK}/figures/cm_{TASK}_{MODEL}_base.png")
    print(f"  [base] {TASK} acc {m['accuracy']:.4f}", flush=True)

def finish_current_only(state, strategy, model):
    m = _eval(model, TASK); _record(state, strategy, TASK, m, _stage_minutes(state))
    print(f"  [{strategy}] {TASK} acc {m['accuracy']:.4f}", flush=True)


# ===== CELL 7 — driver: restore/init, then run the stage plan with resume =====
SESSION_START = time.time()
for sub in ("models", "csv", "xlsx", "docx", "figures", "resume"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)

# Loud banner so a wrong TASK/RESUME in CELL 2 is obvious BEFORE hours of compute.
_mode = ("RESUME (continue an unfinished task)" if RESUME
         else ("FRESH BASE — trains a NEW T1 KAN from scratch" if IS_BASE
               else "FRESH START of this CL task (its first session)"))
print("\n" + "#" * 68 + f"\n#  ABOUT TO RUN:  TASK={TASK}  |  {_mode}\n" + "#" * 68, flush=True)
if not RESUME and not IS_BASE:
    print("#  (RESUME=False on a CL task = start it over. If you meant to CONTINUE, set RESUME=True.)", flush=True)

if RESUME:
    STATE = restore_work_from_resume()          # picks the resume file whose task == TASK
    assert STATE["task"] == TASK, f"resume file is for {STATE['task']}, not {TASK}"
    _restore_rng(STATE.get("rng"))
    print(f"RESUMED {TASK}: stage {STATE['stage_idx']} epoch {STATE['epoch']} | done: {STATE['completed']}", flush=True)
else:
    STATE = init_state()

epoch_logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_{TASK}.csv")
session_expert = load_expert() if not IS_BASE else None
PLAN = stage_plan()
print("stage plan:", PLAN, flush=True)

try:
    # expert ceiling on prior tasks (CL only, once) = reference row of R
    if not IS_BASE and not STATE["expert_done"]:
        for p in PRIOR:
            m = _eval(session_expert, p); STATE["expert_acc"][p] = m["accuracy"]
            _record(STATE, "expert_before", p, m)
            print(f"  expert on {p}: acc {m['accuracy']:.4f}", flush=True)
        STATE["expert_done"] = True; save_resume(STATE)

    while STATE["stage_idx"] < len(PLAN):
        si = STATE["stage_idx"]; strat = PLAN[si]
        resuming_here = (STATE["epoch"] > 0 and STATE["strategy"] == strat)

        fisher_list = star_list = None
        if strat == "ewc":
            fpath = f"{WORK}/resume/fisher_{TASK}_{MODEL}.pt"
            if not (STATE["fisher_ready"] and os.path.exists(fpath)):
                fresh = load_expert()                          # fresh expert (LwF may have frozen the session one)
                fl, sl = [], []
                for p in PRIOR:
                    fi, st = sb.compute_fisher(fresh, prior_train[p], DEVICE, max_batches=EWC_FISHER_BATCHES)
                    fl.append({k: v.cpu() for k, v in fi.items()}); sl.append({k: v.cpu() for k, v in st.items()})
                torch.save({"fisher": fl, "star": sl}, fpath)
                STATE["fisher_ready"] = True; save_resume(STATE)
                del fresh
            fd = torch.load(fpath, map_location=DEVICE, weights_only=False)
            fisher_list = [{k: v.to(DEVICE) for k, v in f.items()} for f in fd["fisher"]]
            star_list   = [{k: v.to(DEVICE) for k, v in s.items()} for s in fd["star"]]

        model = build_stage_model(strat, STATE["latest_model"] if resuming_here else None)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        if resuming_here and STATE.get("latest_optim"):
            optimizer.load_state_dict(STATE["latest_optim"])
        cfg = stage_config(strat, fisher_list, star_list)
        model = run_stage(strat, model, optimizer, STATE["epoch"] if resuming_here else 0, cfg, STATE)

        if strat in ("naive", "rehearsal", "lwf", "ewc"):
            finish_cl_stage(STATE, strat, model)
        elif strat == "base":
            finish_base(STATE, model)
        else:
            finish_current_only(STATE, strat, model)

        STATE.update({"stage_idx": si + 1, "epoch": 0, "strategy": None, "latest_model": None,
                      "latest_optim": None, "best_avg": -1.0, "best_state": None, "history": []})
        STATE["completed"].append(strat)
        save_resume(STATE)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    STATE["task_done"] = True; save_resume(STATE)
    print("\n*** TASK COMPLETE ***", flush=True)
except _Budget:
    print("\n*** SESSION BUDGET REACHED — resume next session (see CELL 8) ***", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\n[ERROR] stage failed: {e}\nResume state saved — fix, then re-run with RESUME=True.", flush=True)


# ===== CELL 8 — write outputs from STATE, bundle, and print next step =====
def _safe(step, fn):
    try:
        fn(); print(f"  wrote {step}")
    except Exception as e:
        print(f"  [warn] {step} failed: {e}")

_safe("final_metrics.csv", lambda: sc.write_final_metrics_csv(STATE["final_rows"], f"{WORK}/csv/final_metrics_{TASK}.csv"))
_safe("perclass.csv",      lambda: sc.write_perclass_csv(STATE["perclass_rows"], f"{WORK}/csv/perclass_{TASK}.csv"))
_safe("clmetrics.csv",     lambda: sc.write_final_metrics_csv(STATE["clmetrics_rows"], f"{WORK}/csv/clmetrics_{TASK}.csv"))
_safe("cl_matrices.json",  lambda: json.dump({"seq": SEQ, "matrices": STATE["cl_matrices"]},
                                             open(f"{WORK}/csv/cl_matrices_{TASK}.json", "w"), indent=2))
_safe("results.xlsx",      lambda: sc.build_results_xlsx(
    {"final_metrics": STATE["final_rows"], "cl_metrics": STATE["clmetrics_rows"], "per_class": STATE["perclass_rows"]},
    f"{WORK}/xlsx/{TASK}_results.xlsx"))
_safe("report.docx",       lambda: sc.build_report_docx(TASK, STATE["clmetrics_rows"], f"{WORK}/docx/{TASK}_report.docx",
                                                        figures_dir=f"{WORK}/figures", title=f"STARL v2 — {TASK} KAN"))
zip_path = bundle()
print("\nBUNDLED ->", zip_path)
if STATE["task_done"]:
    print(f"{TASK} KAN done. Upload {os.path.basename(zip_path)} as a dataset; it holds "
          f"expert_{TASK}_{MODEL}.pth for the next task (or is the final KAN result at T4).")
else:
    print(f"{TASK} NOT finished (stage {STATE['stage_idx']}/{len(PLAN)}, done={STATE['completed']}).\n"
          f"To continue: download {os.path.basename(zip_path)} -> upload as a Kaggle dataset -> attach it -> "
          f"set RESUME=True -> Save & Run All.")
import pandas as pd
if STATE["clmetrics_rows"]:
    print("\n=== CL METRICS so far ===")
    print(pd.DataFrame(STATE["clmetrics_rows"]).to_string(index=False))
