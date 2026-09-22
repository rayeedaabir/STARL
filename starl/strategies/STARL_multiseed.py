# =============================================================================
# STARL v2 — MULTI-SEED VARIANCE RUNNER  (professor #13: "run it multiple times")
# =============================================================================
# Runs the ENTIRE T1->T2->T3->T4 continual chain for ONE seed, for several models,
# inside a SINGLE notebook — no downloading/re-uploading expert zips between tasks.
# Run it once per extra seed (123, then 2025) and combine with the existing seed-42
# results to get mean +/- std error bars on the headline numbers.
#
# STRATEGIES: naive + rehearsal only (configurable). Rationale for the paper: the
# headline claim is rehearsal's retention and naive is its lower bound, so those are
# the two numbers that need error bars. LwF/EWC collapse to ~0 and frozen-probe /
# independent are single-task references — variance there costs 3x the GPU time and
# adds little. This matches standard practice: report variance on the main method.
#
# RESUME (same machinery as the KAN runner, which is battle-tested):
#   * stops CLEANLY after SESSION_MAX_MINUTES so a Commit still saves its output
#   * checkpoints every RESUME_EVERY_EPOCHS epochs AND at every stage boundary
#   * to continue: download {SEED}_multiseed_outputs.zip -> upload as a dataset ->
#     attach it -> set RESUME=True -> Save & Run All
#
# ATTACH: starl-code, starl-splits, ALL FOUR image datasets, and (when RESUME=True)
# the uploaded multiseed output zip. No expert zips needed — T1 is trained here.


# ===== CELL 1 — imports =====
import sys, os, json, time, glob, copy, shutil, zipfile
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
for pkg, mod in (("python-docx", "docx"),):
    try: __import__(mod)
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])


# ===== CELL 2 — config =====
SEED   = 123            # <-- run once with 123, then again with 2025
RESUME = False          # True to continue this seed's unfinished run
RESUME_DIR = None       # None = auto-find this seed's resume_state.pt under /kaggle/input

SESSION_MAX_MINUTES  = 660     # stop cleanly before Kaggle's ~12h cap
RESUME_EVERY_EPOCHS  = 5       # checkpoint cadence (heavy models write ~0.5GB/save)

EPOCHS, LR, BATCH_SIZE, IMAGE_SIZE, WORKERS = 15, 1e-4, 32, 224, 4
USE_AMP = True
MODELS_TO_RUN = ["resnet18", "resnet50", "resnet152", "vgg16", "swin_tiny", "coatnet_0"]
STRATEGIES    = ["naive", "rehearsal"]     # rehearsal propagates the chain; naive branches
REHEARSAL_FRACTION, REH_BATCH = 0.10, 4    # identical to the seed-42 main runs

WORK = f"/kaggle/working/MULTISEED_seed{SEED}"
for sub in ("models", "csv", "xlsx", "docx", "figures", "resume"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_224"
RESUME_ZIP = f"/kaggle/working/seed{SEED}_multiseed_outputs.zip"
sc.set_seed(SEED)
DEVICE = sc.get_device()
print(f"SEED={SEED} | RESUME={RESUME} | models={MODELS_TO_RUN} | strategies={STRATEGIES}")


# ===== CELL 3 — class space, path resolver, loaders =====
cs    = sc.load_class_space(SPLITS_DIR)
TASKS = list(cs.task_order)
NT    = len(TASKS)

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
        print("  building /kaggle/input image index...", flush=True)
        _INPUT_IDX = _build_input_index(); print(f"  indexed {len(_INPUT_IDX)} images", flush=True)
    return _INPUT_IDX.get(os.path.basename(path), path)

train_tf, eval_tf = sc.build_transforms(IMAGE_SIZE)
def _rows(task, split): return sc.read_manifest(sc.manifest_path(SPLITS_DIR, task), split)
def _mk(rows, tf, batch, shuffle):
    return sc._loader(rows, tf, batch, shuffle, WORKERS, path_remap=REMAP, cache_dir=CACHE_DIR)

for t in TASKS:
    assert os.path.exists(REMAP(_rows(t, "train")[0][0])), f"{t}: images missing — attach all 4 datasets."
TRAIN_L = {t: _mk(_rows(t, "train"), train_tf, BATCH_SIZE, True)  for t in TASKS}
VAL_L   = {t: _mk(_rows(t, "val"),   eval_tf,  BATCH_SIZE, False) for t in TASKS}
TEST_L  = {t: _mk(_rows(t, "test"),  eval_tf,  BATCH_SIZE, False) for t in TASKS}
REH_L   = {}
for t in TASKS[:-1]:
    REH_L[t], _n = sc.build_rehearsal_loader(SPLITS_DIR, t, fraction=REHEARSAL_FRACTION,
                                             reh_batch=REH_BATCH, image_size=IMAGE_SIZE,
                                             workers=WORKERS, seed=SEED, path_remap=REMAP)
    print(f"  rehearsal[{t}] = {_n} images")
print("  loaders ready |", {t: len(TRAIN_L[t].dataset) for t in TASKS})


# ===== CELL 4 — the per-model stage plan + resume machinery =====
# One model's plan: T1 base, then for each later task a naive branch and a rehearsal step.
def stage_plan():
    plan = [(TASKS[0], "base")]
    for t in TASKS[1:]:
        for s in STRATEGIES:
            plan.append((t, s))
    return plan
PLAN = stage_plan()
print("per-model plan:", PLAN)

def _snap_rng():
    return {"torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "np": np.random.get_state()}
def _restore_rng(r):
    if not r: return
    try:
        torch.set_rng_state(r["torch"].cpu().to(torch.uint8))
        if r.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in r["cuda"]])
        np.random.set_state(r["np"])
    except Exception as e:
        print(f"  [warn] RNG not restored ({e}) — continuing.", flush=True)

def bundle():
    return sc.bundle_outputs(f"seed{SEED}", WORK, out_zip=RESUME_ZIP, run_manifest=RUN_MANIFEST)

def save_resume(state, rezip=True):
    os.makedirs(f"{WORK}/resume", exist_ok=True)
    tmp = f"{WORK}/resume/resume_state.pt.tmp"
    torch.save(state, tmp); os.replace(tmp, f"{WORK}/resume/resume_state.pt")
    if rezip: bundle()

def _load_state(pt):
    return torch.load(pt, map_location="cpu", weights_only=False)   # our own file; holds RNG state

def restore_from_resume():
    """Find THIS SEED's resume bundle (every finished seed's zip also has a resume file)."""
    base = RESUME_DIR or "/kaggle/input"
    found, matches = {}, []
    for pt in glob.glob(f"{base}/**/resume_state.pt", recursive=True):
        try: st = _load_state(pt)
        except Exception as e: print(f"  [warn] unreadable {pt}: {e}"); continue
        found[st.get("seed")] = found.get(st.get("seed"), 0) + 1
        if st.get("seed") == SEED: matches.append((os.path.dirname(os.path.dirname(pt)), st))
    if matches:
        matches.sort(key=lambda rs: rs[1].get("done", False))      # prefer an UNFINISHED one
        root, _ = matches[0]
        for item in os.listdir(root):
            s, d = os.path.join(root, item), os.path.join(WORK, item)
            if os.path.isdir(s): shutil.copytree(s, d, dirs_exist_ok=True)
            else: os.makedirs(WORK, exist_ok=True); shutil.copy2(s, d)
        return _load_state(f"{WORK}/resume/resume_state.pt")
    zips = glob.glob(f"{base}/**/seed{SEED}_multiseed_outputs.zip", recursive=True)
    if zips:
        with zipfile.ZipFile(zips[0]) as z: z.extractall(WORK)
        return _load_state(f"{WORK}/resume/resume_state.pt")
    raise FileNotFoundError(
        f"RESUME=True for seed {SEED} but no matching resume_state.pt found under {base}.\n"
        f"  Resume files present are for seeds: {dict(found) or 'none'}.\n"
        f"  Attach THIS seed's own output zip (seed{SEED}_multiseed_outputs.zip).")

def init_state():
    return {"seed": SEED, "model_idx": 0, "stage_idx": 0, "epoch": 0, "strategy": None,
            "latest_model": None, "latest_optim": None, "best_avg": -1.0, "best_state": None,
            "history": [], "chain_expert": None, "chain_task": None,
            "R": {}, "final_rows": [], "clmetrics_rows": [], "completed": [],
            "done": False, "rng": None}

RUN_MANIFEST = {"experiment": "multiseed", "seed": SEED, "models": MODELS_TO_RUN,
                "strategies": STRATEGIES, "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                "rehearsal_fraction": REHEARSAL_FRACTION, "reh_batch": REH_BATCH, "tasks": TASKS}


# ===== CELL 5 — model construction + the epoch loop with budget/checkpointing =====
class _Budget(Exception): pass

def build_stage_model(name, task, strategy, state, resuming):
    """base = fresh ImageNet model at T1 head size; later stages grow the head and inherit
    the maintained (rehearsal) chain expert's backbone."""
    num = cs.num_classes_through(task)
    if strategy == "base":
        m = sc.build_model(name, num, DEVICE, SEED, pretrained=True, feature_extract=True)
    else:
        m = sc.build_model(name, num, DEVICE, SEED, pretrained=False, feature_extract=True)
        prev = sc.build_model(name, cs.num_classes_through(state["chain_task"]), DEVICE, SEED,
                              pretrained=False, feature_extract=True)
        prev.load_state_dict(state["chain_expert"])
        m, _ = sc.transfer_backbone(prev, m, name)
        del prev
    if resuming and state.get("latest_model") is not None:
        m.load_state_dict(state["latest_model"])
    return m

def run_stage(name, task, strategy, model, optimizer, start_epoch, state):
    """Train one (model, task, strategy) stage; checkpoint periodically; honour the budget."""
    criterion = torch.nn.CrossEntropyLoss()
    seen = TASKS[:TASKS.index(task) + 1]
    eval_loaders = {t: VAL_L[t] for t in seen}
    reh_loaders = [REH_L[t] for t in TASKS[:TASKS.index(task)]] if strategy == "rehearsal" else None
    use_amp = USE_AMP and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if start_epoch > 0:
        best_avg, best_state, history = state["best_avg"], state["best_state"], state["history"]
    else:
        best_avg, best_state, history = -1.0, None, []
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.time(); model.train(); run = correct = total = 0.0
        if strategy == "rehearsal":
            iters = [iter(r) for r in reh_loaders]
            for s_img, s_lbl in TRAIN_L[task]:
                imgs, lbls = [s_img], [s_lbl]
                for i, rl in enumerate(reh_loaders):
                    try: r_i, r_l = next(iters[i])
                    except StopIteration: iters[i] = iter(rl); r_i, r_l = next(iters[i])
                    imgs.append(r_i); lbls.append(r_l)
                images = torch.cat(imgs).to(DEVICE); labels = torch.cat(lbls).to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(images); loss = criterion(out, labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                run += loss.item() * images.size(0)
                _, pr = torch.max(out, 1); correct += (pr == labels).sum().item(); total += images.size(0)
        else:
            for images, labels in TRAIN_L[task]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(images); loss = criterion(out, labels)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                run += loss.item() * images.size(0)
                _, pr = torch.max(out, 1); correct += (pr == labels).sum().item(); total += images.size(0)
        accs = {t: sc.quick_accuracy(model, ld, DEVICE) for t, ld in eval_loaders.items()}
        avg = float(np.mean(list(accs.values())))
        row = {"seed": SEED, "model": name, "task": task, "experiment": strategy, "epoch": epoch,
               "time_s": round(time.time() - t0, 2), "train_loss": run / total, "train_acc": correct / total,
               **{f"{t}_val_acc": accs[t] for t in seen}}
        epoch_logger.log_epoch(row); history.append(row)
        print(f"  [{name}/{task}/{strategy}] ep {epoch}/{EPOCHS} {row['time_s']:.0f}s "
              f"loss {run/total:.4f} | val " + " ".join(f"{t}:{accs[t]:.3f}" for t in seen), flush=True)
        if avg > best_avg:
            best_avg, best_state = avg, copy.deepcopy(model.state_dict())
        state.update({"strategy": strategy, "epoch": epoch, "best_avg": best_avg,
                      "best_state": best_state, "history": history,
                      "latest_model": model.state_dict(), "latest_optim": optimizer.state_dict(),
                      "rng": _snap_rng()})
        over_budget = (time.time() - SESSION_START) / 60.0 > SESSION_MAX_MINUTES
        if epoch % RESUME_EVERY_EPOCHS == 0 or epoch == EPOCHS or over_budget:
            save_resume(state)
        if over_budget:
            print(f"  [budget] {SESSION_MAX_MINUTES} min reached at {name}/{task}/{strategy} "
                  f"ep {epoch}; resume saved.", flush=True)
            raise _Budget()
    if best_state is not None: model.load_state_dict(best_state)
    return model


# ===== CELL 6 — driver: model -> stage -> epoch, with full resume =====
SESSION_START = time.time()
if RESUME:
    STATE = restore_from_resume()
    assert STATE["seed"] == SEED, f"resume file is for seed {STATE['seed']}, not {SEED}"
    _restore_rng(STATE.get("rng"))
    print(f"RESUMED seed {SEED}: model {STATE['model_idx']} stage {STATE['stage_idx']} "
          f"epoch {STATE['epoch']} | done: {STATE['completed']}", flush=True)
else:
    STATE = init_state()
epoch_logger = sc.MetricLogger(f"{WORK}/csv/epoch_history_seed{SEED}.csv")

print("\n" + "#" * 70 + f"\n#  MULTI-SEED RUN — seed {SEED} — {'RESUME' if RESUME else 'FRESH'}\n" + "#" * 70, flush=True)

def cl_metrics(R):
    fin = [R[i][NT - 1] for i in range(NT) if R[i][NT - 1] is not None]
    bwt = [R[i][NT - 1] - R[i][i] for i in range(NT - 1)
           if R[i][NT - 1] is not None and R[i][i] is not None]
    fm = [max(v for v in (R[i][j] for j in range(i, NT)) if v is not None) - R[i][NT - 1]
          for i in range(NT - 1) if R[i][NT - 1] is not None]
    return {"ACC": float(np.mean(fin)) if fin else float("nan"),
            "BWT": float(np.mean(bwt)) if bwt else float("nan"),
            "FM": float(np.mean(fm)) if fm else float("nan")}

try:
    while STATE["model_idx"] < len(MODELS_TO_RUN):
        name = MODELS_TO_RUN[STATE["model_idx"]]
        STATE["R"].setdefault(name, {s: [[None] * NT for _ in range(NT)] for s in STRATEGIES})
        while STATE["stage_idx"] < len(PLAN):
            task, strategy = PLAN[STATE["stage_idx"]]
            ti = TASKS.index(task)
            resuming = (STATE["epoch"] > 0 and STATE["strategy"] == strategy)
            model = build_stage_model(name, task, strategy, STATE, resuming)
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
            if resuming and STATE.get("latest_optim"): opt.load_state_dict(STATE["latest_optim"])
            model = run_stage(name, task, strategy, model, opt, STATE["epoch"] if resuming else 0, STATE)
            # ---- evaluate on the frozen TEST split of every task seen so far ----
            for i, t in enumerate(TASKS[:ti + 1]):
                m = sc.evaluate_on_task(model, TEST_L[t], cs.task_idx(t), cs.all_classes, DEVICE)
                STATE["final_rows"].append({"seed": SEED, "model": name, "experiment": strategy,
                                            "trained_through": task, "eval_task": t,
                                            "accuracy": m["accuracy"], "f1_weighted": m["f1_weighted"],
                                            "f1_macro": m["f1_macro"]})
                if strategy == "base":          # T1 start is shared by every strategy's matrix
                    for s in STRATEGIES: STATE["R"][name][s][i][ti] = m["accuracy"]
                else:
                    STATE["R"][name][strategy][i][ti] = m["accuracy"]
            # ---- rehearsal (and the T1 base) carries the chain forward ----
            if strategy in ("base", "rehearsal"):
                STATE["chain_expert"] = {k: v.cpu() for k, v in model.state_dict().items()}
                STATE["chain_task"] = task
                sc.save_checkpoint(model, f"{WORK}/models/expert_{task}_{name}_seed{SEED}.pth",
                                   cs.all_classes, seed=SEED, extra={"stage": strategy})
            STATE.update({"stage_idx": STATE["stage_idx"] + 1, "epoch": 0, "strategy": None,
                          "latest_model": None, "latest_optim": None, "best_avg": -1.0,
                          "best_state": None, "history": []})
            save_resume(STATE)
            del model, opt
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # ---- model finished: compute its CL metrics for each strategy ----
        for s in STRATEGIES:
            met = cl_metrics(STATE["R"][name][s])
            STATE["clmetrics_rows"].append({"seed": SEED, "model": name, "strategy": s, **met})
            print(f"  == {name}/{s}: ACC {met['ACC']:.4f} BWT {met['BWT']:.4f} FM {met['FM']:.4f}", flush=True)
        STATE["completed"].append(name)
        STATE.update({"model_idx": STATE["model_idx"] + 1, "stage_idx": 0,
                      "chain_expert": None, "chain_task": None})
        save_resume(STATE)
    STATE["done"] = True; save_resume(STATE)
    print("\n*** SEED COMPLETE ***", flush=True)
except _Budget:
    print("\n*** SESSION BUDGET REACHED — resume next session (CELL 7 shows how) ***", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"\n[ERROR] {e}\nResume state saved — fix, then re-run with RESUME=True.", flush=True)


# ===== CELL 7 — write outputs, bundle, and print the next step =====
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")

_safe("final_metrics.csv", lambda: pd.DataFrame(STATE["final_rows"]).to_csv(
    f"{WORK}/csv/final_metrics_seed{SEED}.csv", index=False))
_safe("clmetrics.csv", lambda: pd.DataFrame(STATE["clmetrics_rows"]).to_csv(
    f"{WORK}/csv/clmetrics_seed{SEED}.csv", index=False))
_safe("R_matrices.json", lambda: json.dump({"seed": SEED, "tasks": TASKS, "R": STATE["R"]},
                                           open(f"{WORK}/csv/R_matrices_seed{SEED}.json", "w"), indent=2))
_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"cl_metrics": STATE["clmetrics_rows"], "final_metrics": STATE["final_rows"]},
    f"{WORK}/xlsx/seed{SEED}_results.xlsx"))
zip_path = bundle()
print("\nBUNDLED ->", zip_path)
if STATE["done"]:
    print(f"Seed {SEED} COMPLETE. Upload this zip as a dataset; run the aggregation/variance step\n"
          f"once you also have the other seed. Then combine with the seed-42 results for mean±std.")
else:
    print(f"Seed {SEED} NOT finished (model {STATE['model_idx']}/{len(MODELS_TO_RUN)}, "
          f"stage {STATE['stage_idx']}/{len(PLAN)}, done={STATE['completed']}).\n"
          f"To continue: download {os.path.basename(zip_path)} -> upload as a Kaggle dataset -> "
          f"attach it -> set RESUME=True -> Save & Run All.")
if STATE["clmetrics_rows"]:
    print(f"\n=== SEED {SEED} CL METRICS so far ===")
    print(pd.DataFrame(STATE["clmetrics_rows"]).round(4).to_string(index=False))
