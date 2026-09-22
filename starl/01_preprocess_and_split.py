"""
STARL v2 — Phase 1: De-duplication + frozen 70/15/15 split builder
==================================================================
Sequence: OPTION 1a  (APTOS 5-grade base -> ODIR non-DR -> LAG -> HAM10000)
Unified class space: 19 classes,  5 -> 12 -> 12 -> 19 across the four tasks.

WHAT THIS DOES (and does NOT):
  * It does NOT train anything and does NOT copy images.
  * It maps every dataset into the unified label space, perceptual-hash
    de-duplicates ACROSS tasks (earliest task in the sequence wins), then writes
    FROZEN, stratified 70/15/15 train/val/test split *manifests* (CSV lists of
    filepath -> split). Every downstream notebook reads images via these
    manifests, so no image can ever leak across train/val/test or across tasks.
  * Outputs are tiny (CSV/JSON) and bundled into  starl_splits.zip  for download
    and re-upload as a Kaggle dataset the training notebooks depend on.

RUN ON KAGGLE:
  1. Attach the datasets listed in DATASET_ROOTS below (set each path).
  2. Run all. Download /kaggle/working/starl_splits.zip from the Output tab.
  3. Upload it as a Kaggle dataset (e.g. "starl-splits") — training notebooks
     load the frozen manifests from there.

Dependencies (all preinstalled on Kaggle): Pillow, imagehash, pandas, scikit-learn, numpy.
"""

import os, sys, json, zipfile, hashlib, warnings, datetime as dt
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- #
# CONFIG — EDIT THESE PATHS TO MATCH YOUR ATTACHED KAGGLE DATASETS
# ----------------------------------------------------------------------------- #
RANDOM_SEED       = 42
SPLIT_FRACS       = (0.70, 0.15, 0.15)          # train / val / test
PHASH_SIZE        = 8                            # 8x8 -> 64-bit perceptual hash
PHASH_THRESHOLD   = 5                            # Hamming <= this  => near-duplicate
IMG_EXTS          = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm")
OUTPUT_DIR        = "/kaggle/working/starl_splits"

# Each root points at the FOLDER that directly contains the dataset.
# Set to None if a dataset is not attached yet (the script will tell you).
DATASET_ROOTS = {
    # APTOS: any layout works — one train.csv, OR train_1.csv/valid.csv/test.csv with
    # train_images/ val_images/ test_images/ folders. All labelled rows are POOLED and
    # re-split by us (70/15/15) so every task uses the same protocol. Set to your folder.
    "APTOS":     "/kaggle/input/datasets/mariaherrerot/aptos2019",
    # ODIR: your preprocessed class-folder dataset
    "ODIR":      "/kaggle/input/datasets/rayeedaabir/odir-preprocessed/organized_odir_dataset",
    # LAG: your (private) authors' version — non_glaucoma/ & suspicious_glaucoma/ each
    # containing attention_map/ image/ label/  (we read the image/ folder only)
    "LAG":       "/kaggle/input/datasets/rayeedaabir/lag-database-part1-uk/LAG_database_part1_UK",
    # HAM10000: your preprocessed class-folder dataset
    "HAM":       "/kaggle/input/datasets/rayeedaabir/ham10000-preprocessed/organized_ham10000",
    # Messidor-2 is NOT needed for Phase 1 (it is only the final external DR test, later).
    "MESSIDOR2": None,
}

# ----------------------------------------------------------------------------- #
# UNIFIED CLASS SPACE (Option 1a) — fixed, deterministic order
# ----------------------------------------------------------------------------- #
DR_GRADES   = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]   # T1 (APTOS)
ODIR_NONDR  = ["Normal", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]  # T2
HAM_CLASSES = ["Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis",
               "Dermatofibroma", "Melanocytic Nevi", "Melanoma", "Vascular Lesions"]  # T4
# LAG (T3) adds no new classes -> maps into {Normal, Glaucoma}.

ALL_CLASSES = DR_GRADES + ODIR_NONDR + HAM_CLASSES          # 5 + 7 + 7 = 19
LABEL2IDX   = {c: i for i, c in enumerate(ALL_CLASSES)}

# Which unified classes each task is allowed to contribute / be evaluated on:
TASK_CLASSES = {
    "T1_APTOS": DR_GRADES,
    "T2_ODIR":  ODIR_NONDR,
    "T3_LAG":   ["Normal", "Glaucoma"],
    "T4_HAM":   HAM_CLASSES,
}
TASK_ORDER = ["T1_APTOS", "T2_ODIR", "T3_LAG", "T4_HAM"]     # sequence => dedup priority

# ----------------------------------------------------------------------------- #
# LABEL MAPPERS  (robust to folder-name variants; case/underscore-insensitive)
# ----------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()

APTOS_GRADE = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR", 4: "Proliferative DR"}

def map_odir(folder: str):
    t = _norm(folder)
    if t.startswith("normal") or t == "n":                       return "Normal"
    if "glaucoma" in t:                                          return "Glaucoma"
    if "cataract" in t:                                          return "Cataract"
    if "myop" in t:                                              return "Myopia"
    if "hypertens" in t:                                         return "Hypertension"
    if "macular" in t or t in ("amd", "a", "age related macular degeneration"):  return "AMD"
    if t.startswith("other") or t == "o":                       return "Other"
    if "diab" in t or t == "d":                                 return None   # DROP ODIR diabetes (1a)
    return None                                                                # unknown -> drop

def map_lag(folder: str):
    t = _norm(folder)
    if t.startswith("non"):        return "Normal"      # non_glaucoma -> Normal
    if "suspicious" in t or "glaucoma" in t:  return "Glaucoma"
    return None

def map_ham(folder: str):
    t = _norm(folder)
    table = {
        "akiec": "Actinic Keratoses", "actinic keratoses": "Actinic Keratoses",
        "bcc": "Basal Cell Carcinoma", "basal cell carcinoma": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis", "benign keratosis": "Benign Keratosis",
        "benign keratosis like lesions": "Benign Keratosis",
        "df": "Dermatofibroma", "dermatofibroma": "Dermatofibroma",
        "nv": "Melanocytic Nevi", "melanocytic nevi": "Melanocytic Nevi",
        "mel": "Melanoma", "melanoma": "Melanoma",
        "vasc": "Vascular Lesions", "vascular lesions": "Vascular Lesions",
    }
    return table.get(t)

# ----------------------------------------------------------------------------- #
# ITEM + COLLECTORS
# ----------------------------------------------------------------------------- #
@dataclass
class Item:
    task:  str
    path:  str
    label: str
    phash: int = -1
    split: str = ""

def _is_img(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)

def _find_class_dir(root):
    """Descend through single wrapper folders (e.g. odir-preprocessed/ODIR/<classes>)
    until we reach the directory that actually holds the class subfolders."""
    cur = root
    for _ in range(4):
        subs = [d for d in os.listdir(cur) if os.path.isdir(os.path.join(cur, d))]
        has_img = any(_is_img(f) for f in os.listdir(cur))
        if len(subs) != 1 or has_img:
            return cur
        cur = os.path.join(cur, subs[0])
    return cur

def _iter_class_folders(root, mapper, tag=""):
    """Yield (path, label) for a class-folder dataset. Auto-descends wrapper folders,
    walks nested images, and reports any folder names that didn't map (so you can spot
    problems)."""
    root = _find_class_dir(root)
    items, unmapped = [], []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        lab = mapper(entry)
        if lab is None:
            unmapped.append(entry); continue
        for dp, _d, files in os.walk(sub):
            for fn in sorted(files):
                if _is_img(fn):
                    items.append((os.path.join(dp, fn), lab))
    if unmapped:
        print(f"    [{tag}] folders skipped (unmapped or intentionally dropped): {unmapped}")
    return items

def _build_file_index(root):
    """basename (with and without extension) -> full path, over ALL subfolders."""
    idx = {}
    for dp, _d, files in os.walk(root):
        for fn in files:
            if _is_img(fn):
                full = os.path.join(dp, fn)
                idx.setdefault(fn, full)
                idx.setdefault(os.path.splitext(fn)[0], full)
    return idx

def _detect_aptos_cols(df):
    low = {c: str(c).lower() for c in df.columns}
    idcol = diacol = None
    for c in df.columns:
        if low[c] in ("id_code","image","id","filename","image_id","img","name","image_name","id_codes"):
            idcol = c; break
    for c in df.columns:
        if low[c] in ("diagnosis","label","level","grade","dr","class","target","labels"):
            diacol = c; break
    if diacol is None:                      # fallback: a column whose values are all in {0..4}
        for c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v) and set(v.astype(int).unique()) <= set(range(5)) and c != idcol:
                diacol = c; break
    if idcol is None:                       # fallback: first non-diagnosis column
        for c in df.columns:
            if c != diacol: idcol = c; break
    return idcol, diacol

def collect_aptos(root):
    """Pool EVERY labelled row from every CSV in the APTOS folder (train.csv OR
    train_1.csv/valid.csv/test.csv), match ids to image files anywhere under root."""
    csvs = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"APTOS: no CSV found under {root}")
    index = _build_file_index(root)
    out, seen = [], set()
    for csv in csvs:
        df = pd.read_csv(csv)
        idcol, diacol = _detect_aptos_cols(df)
        if diacol is None:
            print(f"    [APTOS] {os.path.basename(csv)}: no label column found — skipped")
            continue
        for _, r in df.iterrows():
            try:
                lab = APTOS_GRADE.get(int(float(r[diacol])))
            except (ValueError, TypeError):
                lab = None
            if lab is None:
                continue
            key = str(r[idcol])
            fp = (index.get(key) or index.get(os.path.splitext(key)[0])
                  or index.get(key + ".png") or index.get(key + ".jpg") or index.get(key + ".jpeg"))
            if fp and fp not in seen:
                out.append((fp, lab)); seen.add(fp)
    return out

def collect_lag(root):
    """LAG (authors' subset): each class folder (non_glaucoma/, suspicious_glaucoma/)
    contains attention_map/ image/ label/ .  Read the fundus images ONLY
    (skip attention_map & label). Works whether images are nested or flat."""
    items = []
    for entry in sorted(os.listdir(root)):
        sub = os.path.join(root, entry)
        if not os.path.isdir(sub):
            continue
        lab = map_lag(entry)
        if lab is None:
            continue
        for dirpath, dirs, files in os.walk(sub):
            base = os.path.basename(dirpath).lower()
            if base in ("attention_map", "label", "labels", "mask", "masks"):
                dirs[:] = []          # do not descend into annotation folders
                continue
            for fn in sorted(files):
                if _is_img(fn):
                    items.append((os.path.join(dirpath, fn), lab))
    return items

def collect(task, root):
    if task == "T1_APTOS": pairs = collect_aptos(root)
    elif task == "T2_ODIR": pairs = _iter_class_folders(root, map_odir, "ODIR")
    elif task == "T3_LAG":  pairs = collect_lag(root)
    elif task == "T4_HAM":  pairs = _iter_class_folders(root, map_ham, "HAM")
    else: raise ValueError(task)
    # deterministic order (sort by path) so splits are reproducible
    pairs = sorted(pairs, key=lambda x: x[0])
    return [Item(task=task, path=p, label=l) for p, l in pairs]

# ----------------------------------------------------------------------------- #
# PERCEPTUAL HASH + BK-TREE NEAR-DUPLICATE DEDUP (earliest task wins)
# ----------------------------------------------------------------------------- #
def compute_phash(path):
    from PIL import Image
    import imagehash
    try:
        with Image.open(path) as im:
            return int(str(imagehash.phash(im.convert("L"), hash_size=PHASH_SIZE)), 16)
    except Exception:
        return None

def _hamming(a, b):
    return bin(a ^ b).count("1")

class BKTree:
    """Minimal BK-tree over 64-bit ints with Hamming distance."""
    def __init__(self): self.tree = None
    def add(self, key, payload):
        if self.tree is None:
            self.tree = [key, payload, {}]; return
        node = self.tree
        while True:
            d = _hamming(key, node[0])
            children = node[2]
            if d in children: node = children[d]
            else: children[d] = [key, payload, {}]; return
    def query(self, key, tol):
        if self.tree is None: return []
        out, stack = [], [self.tree]
        while stack:
            node = stack.pop()
            d = _hamming(key, node[0])
            if d <= tol: out.append((d, node[1]))
            for dist, child in node[2].items():
                if d - tol <= dist <= d + tol: stack.append(child)
        return out

def dedup(items, threshold):
    """Process items in task order; the first occurrence (earliest task) is kept."""
    tree = BKTree()
    kept, removed = [], []
    order = {t: i for i, t in enumerate(TASK_ORDER)}
    items_sorted = sorted(items, key=lambda it: (order[it.task], it.path))
    n_bad = 0
    for it in items_sorted:
        h = compute_phash(it.path)
        if h is None:
            n_bad += 1
            continue
        it.phash = h
        matches = tree.query(h, threshold)
        if matches:
            d, keep = min(matches, key=lambda m: m[0])
            removed.append({"removed_task": it.task, "removed_path": it.path,
                            "removed_label": it.label, "kept_task": keep.task,
                            "kept_path": keep.path, "kept_label": keep.label,
                            "hamming": d})
        else:
            tree.add(h, it)
            kept.append(it)
    return kept, removed, n_bad

# ----------------------------------------------------------------------------- #
# STRATIFIED 70/15/15 SPLIT (per task, stratified by unified label)
# ----------------------------------------------------------------------------- #
def split_items(items, fracs, seed):
    tr, va, te = fracs
    by_task = defaultdict(list)
    for it in items: by_task[it.task].append(it)
    for task, its in by_task.items():
        labels = [it.label for it in its]
        idx = np.arange(len(its))
        # guard: classes with <3 samples can't stratify into 3 splits -> all to train
        cnt = Counter(labels)
        strat = labels if all(cnt[l] >= 3 for l in cnt) else None
        train_idx, temp_idx = train_test_split(
            idx, test_size=(va + te), random_state=seed,
            stratify=(strat if strat is None else [labels[i] for i in idx]))
        temp_labels = [labels[i] for i in temp_idx]
        cnt2 = Counter(temp_labels)
        strat2 = temp_labels if all(cnt2[l] >= 2 for l in cnt2) else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(te / (va + te)), random_state=seed, stratify=strat2)
        for i in train_idx: its[i].split = "train"
        for i in val_idx:   its[i].split = "val"
        for i in test_idx:  its[i].split = "test"
    return items

# ----------------------------------------------------------------------------- #
# OUTPUT WRITERS
# ----------------------------------------------------------------------------- #
def write_outputs(kept, removed, n_bad, out_dir, roots_used):
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"task": it.task, "filepath": it.path, "label": it.label,
             "label_idx": LABEL2IDX[it.label], "split": it.split,
             "phash": format(it.phash, "016x")} for it in kept]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "all_splits.csv"), index=False)
    for task in TASK_ORDER:
        sub = df[df.task == task]
        if len(sub):
            sub.to_csv(os.path.join(out_dir, f"split_manifest_{task}.csv"), index=False)
    pd.DataFrame(removed).to_csv(os.path.join(out_dir, "dedup_report.csv"), index=False)

    manifest = {
        "created": dt.datetime.utcnow().isoformat() + "Z",
        "sequence": "Option-1a  APTOS(5) -> ODIR non-DR(+7) -> LAG(+0) -> HAM(+7)",
        "random_seed": RANDOM_SEED, "split_fracs": SPLIT_FRACS,
        "phash_size": PHASH_SIZE, "phash_threshold": PHASH_THRESHOLD,
        "all_classes": ALL_CLASSES, "label2idx": LABEL2IDX,
        "task_classes": TASK_CLASSES, "task_order": TASK_ORDER,
        "roots_used": roots_used,
        "counts": {"kept": len(kept), "removed_dupes": len(removed),
                   "unreadable": n_bad},
    }
    with open(os.path.join(out_dir, "unified_classes.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return df

def print_summary(df, removed):
    print("\n" + "=" * 74)
    print("SPLIT SUMMARY  (rows = frozen, deduplicated)")
    print("=" * 74)
    for task in TASK_ORDER:
        sub = df[df.task == task]
        if not len(sub):
            print(f"\n{task}: (no data — dataset not attached?)"); continue
        piv = sub.pivot_table(index="label", columns="split",
                              values="filepath", aggfunc="count", fill_value=0)
        for s in ("train", "val", "test"):
            if s not in piv.columns: piv[s] = 0
        piv = piv[["train", "val", "test"]]
        print(f"\n{task}   (total {len(sub)})")
        print(piv.to_string())
    print(f"\nCross-task/near duplicates removed: {len(removed)}")
    if removed:
        pairs = Counter((r["kept_task"], r["removed_task"]) for r in removed)
        for (k, r), n in pairs.most_common():
            print(f"   {n:5d}  removed from {r}  (duplicate of {k})")
    print("=" * 74)

def zip_outputs(out_dir):
    zpath = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, fn), arcname=fn)
    print(f"\nBundled -> {zpath}")
    return zpath

# ----------------------------------------------------------------------------- #
# MAIN
# ----------------------------------------------------------------------------- #
def preflight():
    missing = []
    for task in TASK_ORDER:
        key = task.split("_")[1]
        root = DATASET_ROOTS.get(key)
        if not root or not os.path.isdir(root):
            missing.append((key, root))
    if missing:
        print("MISSING DATASETS — attach these on Kaggle and set their paths in DATASET_ROOTS:")
        for k, r in missing: print(f"   {k:10s}  expected at: {r}")
        print("\nOption-1a needs: APTOS (base), ODIR (organized), LAG (authors' version), HAM (organized).")
        return False
    return True

def main():
    print("STARL v2 — Phase 1 preprocessing (Option 1a)")
    if not preflight():
        sys.exit(1)
    all_items, roots_used = [], {}
    for task in TASK_ORDER:
        key = task.split("_")[1]
        root = DATASET_ROOTS[key]; roots_used[task] = root
        items = collect(task, root)
        print(f"  collected {len(items):6d} images for {task}  from {root}")
        all_items += items
    print(f"\nTotal before dedup: {len(all_items)}. Computing perceptual hashes + de-duplicating...")
    kept, removed, n_bad = dedup(all_items, PHASH_THRESHOLD)
    print(f"Kept {len(kept)} | removed {len(removed)} dupes | {n_bad} unreadable.")
    split_items(kept, SPLIT_FRACS, RANDOM_SEED)
    df = write_outputs(kept, removed, n_bad, OUTPUT_DIR, roots_used)
    print_summary(df, removed)
    zip_outputs(OUTPUT_DIR)
    print("\nDONE. Download starl_splits.zip and upload it as a Kaggle dataset for training notebooks.")

if __name__ == "__main__":
    main()
