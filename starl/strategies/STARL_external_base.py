# =============================================================================
# STARL v2 — BASE-EXPERT EXTERNAL CHECK   (why did DR fail externally?)
# =============================================================================
# Our external validation found that 5-class DR grading does NOT transfer to
# Messidor-2 (accuracy ~= the majority-class baseline). That result has TWO possible
# causes and the paper must not confuse them:
#
#   (A) DOMAIN SHIFT — DR grading never transferred from APTOS to Messidor-2 in the
#       first place (different cameras, population, grading protocol), OR
#   (B) CONTINUAL FORGETTING — it transferred fine originally, but learning T2-T4
#       destroyed the external DR skill.
#
# This notebook settles it with a 2x2 design (pure inference, no training):
#
#                        | APTOS test (in-domain) | Messidor-2 (external)
#   ---------------------+------------------------+----------------------
#   T1 BASE expert       |          A             |          B
#   T4 FINAL expert      |          C             |          D
#
#   A-B  = the domain-shift penalty, measured BEFORE any continual learning
#   A-C  = in-domain forgetting caused by learning T2-T4
#   B-D  = how much continual learning cost the EXTERNAL skill
#   If B is already ~= the majority baseline -> cause (A): domain shift. CL is exonerated.
#   If B is good but D collapses            -> cause (B): CL destroyed external transfer.
#
# ATTACH: starl-code, starl-splits, APTOS, Messidor-2 images + the "MESSIDOR-2 DR
# Grades" dataset, the T1 output zip (expert_T1_APTOS_*.pth) and the T4 output zip(s)
# (expert_T4_HAM_*.pth). Internet ON (pykan, for the KAN model). GPU optional.


# ===== CELL 1 — imports =====
import sys, os, glob, json
CODE_DIR   = "/kaggle/input/starl-code"
SPLITS_DIR = "/kaggle/input/starl-splits"
sys.path.append(CODE_DIR)
import torch, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import starl_core as sc
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
for pkg, mod in (("pykan", "kan"), ("python-docx", "docx")):
    try: __import__(mod)
    except ImportError:
        import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])


# ===== CELL 2 — config =====
SEED, BATCH_SIZE, IMAGE_SIZE, WORKERS = 42, 32, 224, 4
MODELS_TO_RUN = ["resnet18", "resnet50", "resnet152", "swin_tiny", "coatnet_0", "kan"]
WORK = "/kaggle/working/EXTBASE"
for sub in ("csv", "xlsx", "docx", "figures"):
    os.makedirs(f"{WORK}/{sub}", exist_ok=True)
CACHE_DIR = "/kaggle/working/img_cache_ext_224"
sc.set_seed(SEED)
DEVICE = sc.get_device()
cs   = sc.load_class_space(SPLITS_DIR)
ALL  = cs.all_classes
T1   = cs.task_order[0]
T4   = cs.task_order[-1]
DR_IDX = sorted(cs.task_idx(T1))
# The base expert's 5-wide head maps 1:1 onto unified indices only if T1 owns 0..4.
assert DR_IDX == list(range(len(DR_IDX))), (
    f"T1 classes are not unified indices 0..N ({DR_IDX}); the base-head mapping below assumes they are.")
print("device:", DEVICE, "| DR classes:", {i: ALL[i] for i in DR_IDX})


# ===== CELL 3 — Messidor-2 rows (same loader logic as the main external notebook) =====
def read_csv_smart(path):
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] == 1:
            for s in (";", ",", "\t"):
                alt = pd.read_csv(path, sep=s)
                if alt.shape[1] > df.shape[1]: df = alt
        return df
    except Exception:
        return pd.read_csv(path)

def find_grades_csv():
    """ADCIS's messidor-2.csv is only the left/right PAIRING sheet — no grades. Scan every
    attached CSV for one that really has an image column AND a DR-grade column."""
    for p in sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True)):
        try: df = read_csv_smart(p)
        except Exception: continue
        cols = [c.lower().strip() for c in df.columns]
        if any(("dr_grade" in c) or ("adjudicated" in c and "grade" in c) or c in ("grade", "dr_level")
               for c in cols) and any(("image" in c) or c in ("id", "filename", "file") for c in cols):
            print(f"  grades CSV: {p}\n    columns: {list(df.columns)}")
            return df
    return None

_IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"); _IDX = None
def img_index():
    global _IDX
    if _IDX is None:
        _IDX = {}
        for r, _d, files in os.walk("/kaggle/input"):
            for f in files:
                if f.lower().endswith(_IMG_EXT):
                    _IDX.setdefault(f, os.path.join(r, f))
                    _IDX.setdefault(os.path.splitext(f)[0], os.path.join(r, f))
        print(f"  indexed {len(_IDX)} image names")
    return _IDX
def resolve(name):
    idx = img_index()
    for k in (os.path.basename(str(name)), os.path.splitext(os.path.basename(str(name)))[0]):
        if k in idx: return idx[k]
    return None
def pick_col(df, *cands, contains=None):
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower(): return c
    return None

mdf = find_grades_csv()
assert mdf is not None, ("No DR-grade CSV found. ADCIS ships images + a left/right pairing sheet "
                         "only — also attach the Kaggle 'MESSIDOR-2 DR Grades' dataset.")
c_img  = pick_col(mdf, "image_id", "image", "id", "filename", contains="image")
c_gr   = pick_col(mdf, "adjudicated_dr_grade", "dr_grade", "grade", contains="grade")
c_gd   = pick_col(mdf, "adjudicated_gradable", "gradable", contains="gradable")
messidor_rows, miss = [], 0
for _, r in mdf.iterrows():
    if c_gd is not None and str(r[c_gd]).strip() in ("0", "0.0", "False", "false"): continue
    try: g = int(float(r[c_gr]))
    except (ValueError, TypeError): continue
    if not (0 <= g <= 4): continue
    p = resolve(r[c_img])
    if p is None: miss += 1; continue
    messidor_rows.append((p, g))        # grade g == unified index g (asserted in CELL 2)
print(f"  Messidor-2 usable: {len(messidor_rows)} ({miss} unresolved)")

_, eval_tf = sc.build_transforms(IMAGE_SIZE)
def mk(rows): return sc._loader(rows, eval_tf, BATCH_SIZE, False, WORKERS, None, CACHE_DIR)
MESSIDOR_L = mk(messidor_rows)

# in-domain reference: APTOS's own frozen TEST split (needs the APTOS dataset attached)
_IDXG = None
def REMAP(path):
    if os.path.exists(path): return path
    return resolve(path) or path
aptos_rows = [(REMAP(p), l) for p, l in sc.read_manifest(sc.manifest_path(SPLITS_DIR, T1), "test")]
assert os.path.exists(aptos_rows[0][0]), "APTOS test images not found — attach the APTOS dataset."
APTOS_L = mk(aptos_rows)
print(f"  APTOS frozen test: {len(aptos_rows)} images")


# ===== CELL 4 — evaluation restricted to the 5 DR classes (works for 5- and 19-wide heads) =====
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

def eval_dr(model, loader):
    """Argmax restricted to the DR class indices, so a 5-class base head and a 19-class
    final head are compared on exactly the same decision problem."""
    model.eval(); ys, ps = [], []
    allowed = torch.tensor(DR_IDX, device=DEVICE)
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(DEVICE))
            m = torch.full_like(out, float("-inf")); m[:, allowed] = out[:, allowed]
            ps.extend(m.argmax(1).cpu().numpy()); ys.extend(labels.numpy())
    y, p = np.array(ys), np.array(ps)
    _, counts = np.unique(y, return_counts=True)
    return {"accuracy": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "majority_baseline": float(counts.max() / len(y)), "n": int(len(y)),
            "_cm": confusion_matrix(y, p, labels=DR_IDX).tolist()}

def load_expert(model_name, task):
    hits = glob.glob(f"/kaggle/input/**/expert_{task}_{model_name}.pth", recursive=True)
    if not hits: return None
    m = sc.build_model(model_name, cs.num_classes_through(task), DEVICE, SEED,
                       pretrained=False, feature_extract=True)
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.to(DEVICE).eval()
    return m


# ===== CELL 5 — run the 2x2 for every model =====
rows = []
for name in MODELS_TO_RUN:
    print(f"\n=== {name} ===", flush=True)
    for stage, task in (("T1_base_expert", T1), ("T4_final_expert", T4)):
        try:
            model = load_expert(name, task)
            if model is None:
                print(f"  [skip] {name}: no expert_{task}_{name}.pth attached"); continue
            for dsname, ld in (("APTOS_test_internal", APTOS_L), ("Messidor2_external", MESSIDOR_L)):
                r = eval_dr(model, ld)
                rows.append({"model": name, "stage": stage, "dataset": dsname,
                             **{k: v for k, v in r.items() if not k.startswith("_")}})
                print(f"  {stage:<16} {dsname:<20} acc {r['accuracy']:.4f} "
                      f"| balanced {r['balanced_accuracy']:.4f} "
                      f"| F1m {r['f1_macro']:.4f} [majority {r['majority_baseline']:.4f}]", flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            import traceback; print(f"  [ERROR] {name}/{stage}: {e}"); traceback.print_exc()

df = pd.DataFrame(rows)


# ===== CELL 6 — the decisive table: decompose the external failure =====
diag = []
for name in df.model.unique() if not df.empty else []:
    d = df[df.model == name]
    def get(stage, ds, col="balanced_accuracy"):
        s = d[(d.stage == stage) & (d.dataset == ds)]
        return float(s[col].iloc[0]) if len(s) else np.nan
    A = get("T1_base_expert",  "APTOS_test_internal")
    B = get("T1_base_expert",  "Messidor2_external")
    C = get("T4_final_expert", "APTOS_test_internal")
    D = get("T4_final_expert", "Messidor2_external")
    diag.append({"model": name,
                 "A base/in-domain": A, "B base/external": B,
                 "C final/in-domain": C, "D final/external": D,
                 "domain_shift_penalty (A-B)": A - B,
                 "in_domain_forgetting (A-C)": A - C,
                 "external_loss_from_CL (B-D)": B - D})
diag_df = pd.DataFrame(diag)

# Compare the two causes on the SAME scale instead of thresholding raw balanced accuracy.
# (An earlier version tested "is B below chance+0.05", which mislabels a base model that is
# only ~0.09 above chance as though it had real external skill. The right question is what
# FRACTION of the model's above-chance skill each effect destroys.)
CHANCE = 1.0 / len(DR_IDX)
def diagnose(row):
    A, B, D = row["A base/in-domain"], row["B base/external"], row["D final/external"]
    if any(np.isnan(v) for v in (A, B, D)): return "incomplete"
    skill_A = A - CHANCE                      # in-domain skill above chance, before CL
    if skill_A <= 0: return "base model has no in-domain skill — inconclusive"
    kept = (B - CHANCE) / skill_A             # fraction of that skill surviving the domain change
    domain_hit, cl_hit = A - B, B - D
    ratio = domain_hit / cl_hit if cl_hit > 1e-6 else float("inf")
    if kept < 0.35 and ratio >= 2:
        return f"DOMAIN SHIFT dominates (external keeps only {kept*100:.0f}% of in-domain skill; {ratio:.1f}x the CL effect)"
    if kept >= 0.35 and cl_hit > 0.05:
        return f"CONTINUAL LEARNING dominates (external skill was real: {kept*100:.0f}% kept, then lost {cl_hit:.3f})"
    return f"MIXED (external keeps {kept*100:.0f}%; domain hit {domain_hit:.3f} vs CL hit {cl_hit:.3f})"
if not diag_df.empty:
    diag_df["skill_kept_externally_%"] = ((diag_df["B base/external"] - CHANCE) /
                                          (diag_df["A base/in-domain"] - CHANCE) * 100).round(1)
    diag_df["verdict"] = diag_df.apply(diagnose, axis=1)


# ===== CELL 7 — figure, tables, bundle =====
def _safe(step, fn):
    try: fn(); print(f"  wrote {step}")
    except Exception as e: print(f"  [warn] {step}: {e}")
_safe("extbase_raw.csv", lambda: df.to_csv(f"{WORK}/csv/extbase_raw.csv", index=False))
_safe("extbase_diagnosis.csv", lambda: diag_df.to_csv(f"{WORK}/csv/extbase_diagnosis.csv", index=False))
_safe("results.xlsx", lambda: sc.build_results_xlsx(
    {"diagnosis": diag_df.to_dict("records"), "raw": df.to_dict("records")},
    f"{WORK}/xlsx/EXTBASE_results.xlsx"))

def fig_2x2():
    if df.empty: return
    models = list(df.model.unique()); x = np.arange(len(models)); w = 0.2
    fig, ax = plt.subplots(figsize=(2.2 + 1.5 * len(models), 4.6))
    combos = [("T1_base_expert", "APTOS_test_internal", "base · in-domain"),
              ("T1_base_expert", "Messidor2_external", "base · EXTERNAL"),
              ("T4_final_expert", "APTOS_test_internal", "final · in-domain"),
              ("T4_final_expert", "Messidor2_external", "final · EXTERNAL")]
    for k, (stage, ds, lab) in enumerate(combos):
        vals = []
        for m in models:
            s = df[(df.model == m) & (df.stage == stage) & (df.dataset == ds)]
            vals.append(float(s.balanced_accuracy.iloc[0]) if len(s) else np.nan)
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab)
    ax.axhline(1.0 / len(DR_IDX), ls="--", c="k", lw=1, label=f"chance ({1/len(DR_IDX):.2f})")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("balanced accuracy (5-class DR)")
    ax.set_title("Where does external DR performance go? base vs final expert, internal vs external")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{WORK}/figures/extbase_2x2.png", dpi=300); plt.close(fig)
    print("  fig extbase_2x2")
_safe("figure", fig_2x2)
_safe("report.docx", lambda: sc.build_report_docx(
    "EXTBASE", diag_df.to_dict("records"), f"{WORK}/docx/EXTBASE_report.docx",
    figures_dir=f"{WORK}/figures", title="STARL v2 — Base vs final expert, internal vs external DR"))

zip_path = sc.bundle_outputs("EXTBASE", WORK, out_zip="/kaggle/working/EXTBASE_outputs.zip",
                             run_manifest={"experiment": "base_expert_external_check",
                                           "models": MODELS_TO_RUN, "dr_classes": DR_IDX,
                                           "messidor_images": len(messidor_rows),
                                           "aptos_test_images": len(aptos_rows)})
print("\nBUNDLED ->", zip_path)
if not df.empty:
    print("\n=== RAW (balanced accuracy is the honest column; chance = "
          f"{1/len(DR_IDX):.2f}) ===")
    print(df[["model", "stage", "dataset", "n", "majority_baseline", "accuracy",
              "balanced_accuracy", "f1_macro"]].round(4).to_string(index=False))
if not diag_df.empty:
    print("\n=== DIAGNOSIS (balanced accuracy) ===")
    print(diag_df.round(4).to_string(index=False))
    print("\nHOW TO READ: if 'B base/external' is already near chance, DR never transferred "
          "(domain shift)\nand continual learning is exonerated. If B is healthy but D collapsed, "
          "continual learning\ndestroyed the external skill. The verdict column states which.")
