# =============================================================================
# STARL v2 — EXTERNAL VALIDATION  (professor #2 / #3)
# =============================================================================
# Evaluates the FINAL continual models (expert_T4_HAM_*.pth) on datasets they have
# NEVER seen, from different hospitals/populations/devices:
#   * Messidor-2 (1,748 fundus images, DR grades 0-4)  -> validates the T1/DR skill
#   * Derm7pt    (dermoscopic lesion images)           -> validates the T4/skin skill
# This is the strongest possible answer to "does it generalize, or did it just fit
# your splits?" Nothing here is trained — pure inference on held-out external data.
#
# TWO NUMBERS ARE REPORTED PER DATASET (both matter, report both):
#   * open-set  : plain argmax over the FULL 19-class head (deployment-realistic:
#                 the model may answer with a class from another task)
#   * restricted: argmax limited to the classes this external set can contain
#                 (measures the skill itself, isolated from cross-task confusion)
#
# ATTACH: starl-code, starl-splits, the T4 output zip(s) (for expert_T4_HAM_*.pth),
# and the external datasets. GPU recommended (inference only, so it's quick).
#
# CELL 4 PRINTS THE LABEL MAPPING AND STOPS SHORT OF TRAINING — read it and confirm
# it looks right before running the rest.


# ===== CELL 1 — imports =====
import sys, os, glob, json, re
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
# pykan is needed ONLY to rebuild the KAN model (sc.build_model imports `kan`), and
# python-docx for the report. Neither is preinstalled on Kaggle -> internet must be ON.
# If pykan can't install, the KAN model is skipped and the other 11 still run.
for pkg, mod in (("pykan", "kan"), ("python-docx", "docx")):
    try:
        __import__(mod)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])


# ===== CELL 2 — config =====
SEED, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 32, 224, 4
# Which final models to validate externally (any with an expert_T4_HAM_*.pth available):
MODELS_TO_RUN = ["resnet18", "resnet50", "resnet152", "swin_tiny", "coatnet_0", "kan"]
RUN_MESSIDOR = True
RUN_DERM7PT  = True
DERM7PT_MODALITY = "derm"      # "derm" = dermoscopic (matches HAM10000). "clinic" = smartphone-style.

WORK = "/kaggle/working/EXTERNAL"
for sub in ("csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_ext_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs = sc.load_class_space(SPLITS_DIR)
ALL = cs.all_classes
print("device:", DEVICE)
print("unified classes:", {i: c for i, c in enumerate(ALL)})


# ===== CELL 3 — locate the external datasets + read their metadata =====
def _find_dir(*name_parts):
    """Find a directory under /kaggle/input whose path contains all the given parts."""
    hits = []
    for root, dirs, _f in os.walk("/kaggle/input"):
        low = root.lower()
        if all(p.lower() in low for p in name_parts):
            hits.append(root)
    hits.sort(key=len)
    return hits[0] if hits else None

def _find_csv(*name_parts):
    hits = [p for p in glob.glob("/kaggle/input/**/*.csv", recursive=True)
            if all(part.lower() in p.lower() for part in name_parts)]
    hits.sort(key=len)
    return hits[0] if hits else None

def read_csv_smart(path):
    """Read a CSV whatever its delimiter is (ADCIS files are ';'-separated, so a plain
    read_csv collapses them into ONE column named e.g. 'left;right')."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")     # sniff , ; \t |
        if df.shape[1] == 1:                                   # sniffing failed — try ';' then ','
            for s in (";", ",", "\t"):
                alt = pd.read_csv(path, sep=s)
                if alt.shape[1] > df.shape[1]: df = alt
        return df
    except Exception:
        return pd.read_csv(path)

def find_grades_csv():
    """The ADCIS messidor-2.csv is only the LEFT/RIGHT EYE PAIRING sheet — it contains NO
    DR grades. So don't trust the filename: scan every attached CSV and keep the first one
    that actually has both an image column and a DR-grade column."""
    best = None
    for p in sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True)):
        try:
            df = read_csv_smart(p)
        except Exception:
            continue
        cols = [c.lower().strip() for c in df.columns]
        has_grade = any(("dr_grade" in c) or ("adjudicated" in c and "grade" in c)
                        or c in ("grade", "retinopathy_grade", "dr_level") for c in cols)
        has_img = any(("image" in c) or c in ("id", "filename", "file", "img") for c in cols)
        if has_grade and has_img:
            print(f"  found DR-grade CSV: {p}")
            print(f"    columns: {list(df.columns)}")
            return p, df
        if has_grade and best is None:
            best = (p, df)
    return best if best else (None, None)

# index every image once so we can resolve any filename cheaply
_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_IMG_INDEX = None
def img_index():
    global _IMG_INDEX
    if _IMG_INDEX is None:
        _IMG_INDEX = {}
        for r, _d, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower().endswith(_IMG_EXT):
                    _IMG_INDEX.setdefault(f, os.path.join(r, f))
                    _IMG_INDEX.setdefault(os.path.splitext(f)[0], os.path.join(r, f))  # stem too
        print(f"  indexed {len(_IMG_INDEX)} image names")
    return _IMG_INDEX

def resolve_image(name, base_dir=None):
    """Resolve a metadata filename to a real path (tries direct join, then the index)."""
    if base_dir:
        p = os.path.join(base_dir, name)
        if os.path.exists(p): return p
    idx = img_index()
    for key in (os.path.basename(name), os.path.splitext(os.path.basename(name))[0]):
        if key in idx: return idx[key]
    return None

def pick_col(df, *candidates, contains=None):
    """Find a column by exact-ish name, else by substring."""
    low = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low: return low[cand.lower()]
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower(): return c
    return None

MESSIDOR_CSV, MESSIDOR_DF = (find_grades_csv() if RUN_MESSIDOR else (None, None))
DERM_META    = _find_csv("meta") if RUN_DERM7PT else None
if DERM_META and "derm" not in DERM_META.lower():
    DERM_META = _find_csv("derm7pt", "meta") or DERM_META
print("messidor GRADES csv:", MESSIDOR_CSV)
print("derm7pt meta:", DERM_META)
if RUN_MESSIDOR and MESSIDOR_CSV is None:
    print("""
  !! NO DR-GRADE FILE FOUND. This is expected if you only attached the ADCIS download:
     ADCIS ships the IMAGES plus 'messidor-2.csv', which is only the LEFT/RIGHT EYE
     PAIRING sheet — it contains NO diabetic-retinopathy grades (ADCIS states the
     database has no ground truth; grades were published separately).
     FIX: also attach the Kaggle dataset 'MESSIDOR-2 DR Grades' (Google Brain /
     Krause et al., adjudicated by 3 retina specialists). It has image_id +
     adjudicated_dr_grade (+ adjudicated_gradable). Then re-run.
     Meanwhile this notebook will skip Messidor and still do Derm7pt.
""")


# ===== CELL 4 — build (image, unified_label) rows for each external set. REVIEW THE PRINTOUT =====
def _match_class(*keywords, exclude=()):
    """Find the unified class name matching ALL keywords (case-insensitive), if any."""
    for c in ALL:
        lc = c.lower()
        if all(k.lower() in lc for k in keywords) and not any(x.lower() in lc for x in exclude):
            return c
    return None

# ---- Messidor-2: DR grade 0-4 -> the five APTOS DR classes (ordered by severity) ----
DR_ORDER = [
    _match_class("no") or _match_class("none"),                                   # grade 0
    _match_class("mild"),                                                          # 1
    _match_class("moderate"),                                                      # 2
    _match_class("severe", exclude=("prolifer",)),                                 # 3
    _match_class("prolifer"),                                                      # 4
]
# GUARD: if the unified class names differ from what the keyword rules expect, the five DR
# grades could silently collapse onto the same class and quietly corrupt every DR number.
# Fail loudly instead — then set DR_ORDER by hand from the printed class list in CELL 2.
_dr_named = [c for c in DR_ORDER if c is not None]
if RUN_MESSIDOR and MESSIDOR_CSV is not None:
    assert len(_dr_named) == 5 and len(set(_dr_named)) == 5, (
        f"DR grade->class mapping is not 5 distinct classes: {DR_ORDER}. "
        f"Set DR_ORDER manually using the unified class names printed in CELL 2.")

messidor_rows, messidor_classes = [], []
if RUN_MESSIDOR and MESSIDOR_CSV is not None:
    mdf = MESSIDOR_DF if MESSIDOR_DF is not None else read_csv_smart(MESSIDOR_CSV)
    print("\nmessidor columns:", list(mdf.columns))
    c_img   = pick_col(mdf, "image_id", "image", "id", "filename", contains="image")
    c_grade = pick_col(mdf, "adjudicated_dr_grade", "dr_grade", "grade", "diagnosis", contains="grade")
    c_grad  = pick_col(mdf, "adjudicated_gradable", "gradable", contains="gradable")
    print(f"  using image col='{c_img}' grade col='{c_grade}' gradable col='{c_grad}'")
    assert c_img and c_grade, (
        f"That CSV has no usable image/grade columns (found {list(mdf.columns)}).\n"
        "  If these look like 'left'/'right', it is the ADCIS PAIRING sheet, not grades —\n"
        "  attach the Kaggle 'MESSIDOR-2 DR Grades' dataset (image_id + adjudicated_dr_grade).")
    base = _find_dir("messidor", "image") or _find_dir("messidor")
    n_missing = n_ungradable = 0
    for _, r in mdf.iterrows():
        if c_grad is not None and str(r[c_grad]).strip() in ("0", "0.0", "False", "false"):
            n_ungradable += 1; continue                      # skip images graders marked ungradable
        try:
            g = int(float(r[c_grade]))
        except (ValueError, TypeError):
            continue
        if not (0 <= g <= 4) or DR_ORDER[g] is None: continue
        p = resolve_image(str(r[c_img]), base)
        if p is None: n_missing += 1; continue
        messidor_rows.append((p, cs.label2idx[DR_ORDER[g]]))
    messidor_classes = [c for c in DR_ORDER if c is not None]
    print(f"  Messidor-2: {len(messidor_rows)} images usable | {n_missing} unresolved | {n_ungradable} ungradable")
    print("  grade->class:", {i: DR_ORDER[i] for i in range(5)})
    print("  class distribution:", pd.Series([l for _, l in messidor_rows]).map(lambda i: ALL[i]).value_counts().to_dict())

# ---- Derm7pt: diagnosis text -> HAM10000 skin classes ----
# HAM classes: akiec, bcc, bkl (keratosis-like), df, mel, nv, vasc.
# Derm7pt has no akiec; everything else maps by keyword. 'miscellaneous'/'melanosis' are dropped.
SKIN_RULES = [   # (keywords in the derm7pt diagnosis, matcher for the unified class)
    (["basal cell"],            lambda: _match_class("basal") or _match_class("bcc")),
    (["melanoma"],              lambda: _match_class("melanoma", exclude=("nevus",)) or _match_class("mel")),
    (["nevus"],                 lambda: _match_class("nevus") or _match_class("nv") or _match_class("melanocytic")),
    (["seborrheic keratosis"],  lambda: _match_class("keratosis", exclude=("actinic",)) or _match_class("bkl")),
    (["lentigo"],               lambda: _match_class("keratosis", exclude=("actinic",)) or _match_class("bkl")),
    (["dermatofibroma"],        lambda: _match_class("dermatofibroma") or _match_class("df")),
    (["vascular"],              lambda: _match_class("vascular") or _match_class("vasc")),
]
derm_rows, derm_classes = [], []
if RUN_DERM7PT and DERM_META:
    ddf = pd.read_csv(DERM_META)
    print("\nderm7pt columns:", list(ddf.columns))
    c_diag = pick_col(ddf, "diagnosis", "label", contains="diagnos")
    c_path = pick_col(ddf, DERM7PT_MODALITY, contains=DERM7PT_MODALITY)
    print(f"  using diagnosis col='{c_diag}' image col='{c_path}' (modality={DERM7PT_MODALITY})")
    assert c_diag and c_path, "Could not identify Derm7pt diagnosis/image columns — set them by hand."
    base = _find_dir("derm7pt", "images") or _find_dir("images")
    mapped, unmapped, missing = {}, {}, 0
    for _, r in ddf.iterrows():
        diag = str(r[c_diag]).strip().lower()
        target = None
        for keys, matcher in SKIN_RULES:
            if all(k in diag for k in keys):
                target = matcher(); break
        if target is None:
            unmapped[diag] = unmapped.get(diag, 0) + 1; continue
        p = resolve_image(str(r[c_path]), base)
        if p is None: missing += 1; continue
        derm_rows.append((p, cs.label2idx[target]))
        mapped.setdefault(diag, [target, 0]); mapped[diag][1] += 1
    derm_classes = sorted({ALL[l] for _, l in derm_rows})
    print(f"  Derm7pt: {len(derm_rows)} images usable | {missing} unresolved")
    print("  diagnosis -> unified class:")
    for d, (t, n) in sorted(mapped.items()): print(f"    {d:<34} -> {t:<28} ({n})")
    if unmapped:
        print("  DROPPED (no HAM equivalent):", dict(sorted(unmapped.items(), key=lambda kv: -kv[1])))
    print("  covered classes:", derm_classes)

print("\n>>> REVIEW THE MAPPINGS ABOVE BEFORE CONTINUING <<<")


# ===== CELL 5 — evaluation (open-set and restricted) =====
_, eval_tf = sc.build_transforms(IMAGE_SIZE)

def ext_loader(rows):
    return sc._loader(rows, eval_tf, BATCH_SIZE, False, WORKERS, path_remap=None, cache_dir=CACHE_DIR)

def evaluate_external(model, loader, allowed_idx):
    """Returns open-set and restricted metrics + confusion over `allowed_idx`."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                                 confusion_matrix, classification_report, balanced_accuracy_score)
    model.eval(); ys, p_open, p_res = [], [], []
    allowed = torch.tensor(sorted(allowed_idx), device=DEVICE)
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(DEVICE))
            p_open.extend(out.argmax(1).cpu().numpy())
            masked = torch.full_like(out, float("-inf"))
            masked[:, allowed] = out[:, allowed]              # restrict to plausible classes
            p_res.extend(masked.argmax(1).cpu().numpy())
            ys.extend(labels.numpy())
    y = np.array(ys)
    res = {}
    # CONTEXT THAT MUST ACCOMPANY EXTERNAL ACCURACY: external sets are heavily imbalanced
    # (Messidor-2 is ~60% "No DR"), so raw accuracy alone can look fine while the model
    # just predicts the majority class. Report the trivial baseline and balanced accuracy
    # (mean per-class recall, chance = 1/n_classes) beside it.
    _, _counts = np.unique(y, return_counts=True)
    res["majority_class_baseline"] = float(_counts.max() / len(y))
    res["n_classes_present"] = int(len(_counts))
    for tag, p in (("open", np.array(p_open)), ("restricted", np.array(p_res))):
        res[f"accuracy_{tag}"] = accuracy_score(y, p)
        res[f"balanced_accuracy_{tag}"] = balanced_accuracy_score(y, p)
        res[f"f1_weighted_{tag}"] = f1_score(y, p, average="weighted", zero_division=0)
        res[f"f1_macro_{tag}"] = f1_score(y, p, average="macro", zero_division=0)
        res[f"precision_weighted_{tag}"] = precision_score(y, p, average="weighted", zero_division=0)
        res[f"recall_weighted_{tag}"] = recall_score(y, p, average="weighted", zero_division=0)
    names = [ALL[i] for i in sorted(allowed_idx)]
    res["_cm"] = confusion_matrix(y, np.array(p_res), labels=sorted(allowed_idx)).tolist()
    res["_names"] = names
    rep = classification_report(y, np.array(p_res), labels=sorted(allowed_idx),
                                target_names=names, zero_division=0, output_dict=True)
    res["_per_class"] = {c: {"f1": rep[c]["f1-score"], "recall": rep[c]["recall"],
                             "precision": rep[c]["precision"], "support": rep[c]["support"]} for c in names}
    return res

def load_final(model_name):
    hits = glob.glob(f"/kaggle/input/**/expert_T4_HAM_{model_name}.pth", recursive=True)
    if not hits: return None
    m = sc.build_model(model_name, len(ALL), DEVICE, SEED, pretrained=False, feature_extract=True)
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m


# ===== CELL 6 — run external validation for every model =====
def plot_cm(cm, names, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.6 + 0.6 * len(names), 1.4 + 0.6 * len(names)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

DATASETS = []
if messidor_rows: DATASETS.append(("Messidor-2", messidor_rows, [cs.label2idx[c] for c in messidor_classes], "T1_APTOS / DR"))
if derm_rows:     DATASETS.append(("Derm7pt",    derm_rows,     sorted({l for _, l in derm_rows}),           "T4_HAM / skin"))

ext_rows, ext_perclass = [], []
for ds_name, rows, allowed, skill in DATASETS:
    ld = ext_loader(rows)
    print(f"\n=== {ds_name} ({len(rows)} images, {len(allowed)} classes) — validates {skill} ===", flush=True)
    for mname in MODELS_TO_RUN:
        model = None
        try:
            # load INSIDE the try: a model-specific problem (e.g. pykan missing for 'kan')
            # must skip that model, not abort the whole external-validation run.
            model = load_final(mname)
            if model is None:
                print(f"  [skip] {mname}: no expert_T4_HAM_{mname}.pth attached"); continue
            r = evaluate_external(model, ld, allowed)
            ext_rows.append({"dataset": ds_name, "validates": skill, "model": mname, "n_images": len(rows),
                             **{k: v for k, v in r.items() if not k.startswith("_")}})
            for c, v in r["_per_class"].items():
                ext_perclass.append({"dataset": ds_name, "model": mname, "class": c, **v})
            plot_cm(r["_cm"], r["_names"], f"{mname} — {ds_name} (restricted)",
                    f"{WORK}/figures/cm_EXT_{ds_name}_{mname}.png")
            print(f"  {mname}: open {r['accuracy_open']:.4f} | restricted {r['accuracy_restricted']:.4f} "
                  f"| balanced {r['balanced_accuracy_restricted']:.4f} "
                  f"| F1w(res) {r['f1_weighted_restricted']:.4f} "
                  f"[majority-class baseline {r['majority_class_baseline']:.4f}]", flush=True)
        except Exception as e:
            import traceback; print(f"  [ERROR] {mname}: {e}"); traceback.print_exc()
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# ===== CELL 7 — tables, figure, bundle =====
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")

ext_df = pd.DataFrame(ext_rows)
_safe("external_metrics.csv", lambda: ext_df.to_csv(f"{WORK}/csv/external_metrics.csv", index=False))
_safe("external_perclass.csv", lambda: pd.DataFrame(ext_perclass).to_csv(f"{WORK}/csv/external_perclass.csv", index=False))
_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"external_metrics": ext_rows, "external_per_class": ext_perclass}, f"{WORK}/xlsx/EXTERNAL_results.xlsx"))

def fig_summary():
    if ext_df.empty: return
    for ds in ext_df.dataset.unique():
        d = ext_df[ext_df.dataset == ds].sort_values("accuracy_restricted")
        fig, ax = plt.subplots(figsize=(8, 4.4)); y = np.arange(len(d)); h = 0.38
        ax.barh(y - h / 2, d.accuracy_open, h, label="open-set (full 19-class head)")
        ax.barh(y + h / 2, d.accuracy_restricted, h, label="restricted to this domain")
        ax.set_yticks(y); ax.set_yticklabels(d.model)
        ax.set_xlabel("accuracy on external data"); ax.set_xlim(0, 1)
        ax.set_title(f"External validation — {ds} (never seen in training)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{WORK}/figures/external_{ds}.png", dpi=300); plt.close(fig)
        print(f"  fig external_{ds}")
_safe("figures", fig_summary)
_safe("report.docx", lambda: sc.build_report_docx("EXTERNAL", ext_rows, f"{WORK}/docx/EXTERNAL_report.docx",
                                                  figures_dir=f"{WORK}/figures",
                                                  title="STARL v2 — External validation (Messidor-2, Derm7pt)"))

manifest = {"experiment": "external_validation", "datasets": [d[0] for d in DATASETS],
            "models": MODELS_TO_RUN, "derm7pt_modality": DERM7PT_MODALITY,
            "messidor_images": len(messidor_rows), "derm7pt_images": len(derm_rows)}
zip_path = sc.bundle_outputs("EXTERNAL", WORK, out_zip="/kaggle/working/EXTERNAL_outputs.zip", run_manifest=manifest)
print("\nBUNDLED ->", zip_path)
if not ext_df.empty:
    print("\n=== EXTERNAL VALIDATION ===")
    print(ext_df[["dataset", "model", "n_images", "majority_class_baseline",
                  "accuracy_open", "accuracy_restricted", "balanced_accuracy_restricted",
                  "f1_weighted_restricted", "f1_macro_restricted"]].round(4).to_string(index=False))
    print("\nREAD THIS BESIDE THE NUMBERS: compare accuracy_restricted against "
          "majority_class_baseline.\nIf they are close, the model is largely predicting the "
          "majority class — balanced_accuracy\n(chance = 1/n_classes) and f1_macro are then the "
          "honest measures of external skill.")
