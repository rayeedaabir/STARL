# =============================================================================
# STARL — Table 1 split counts  (fills the "[fill]" cells in the Methods table)
# =============================================================================
# The per-task train/val/test counts were never printed into any results zip —
# they live in the Phase-1 split manifests inside the `starl-splits` dataset.
# This reads them directly and prints the table rows, plus the per-class
# distribution needed for Supplementary Table S2.
#
# RUN EITHER:
#   * on Kaggle: new notebook, Accelerator = None, attach ONLY `starl-splits`; or
#   * locally:   unzip starl_splits.zip and set SPLITS_DIR to that folder.
# Takes seconds; no GPU, no torch.

import os, glob, csv, json
from collections import Counter, OrderedDict

SPLITS_DIR = "/kaggle/input/starl-splits"      # local: r"C:\path\to\starl_splits"

# tolerate the folder being nested one level inside the dataset
if not glob.glob(os.path.join(SPLITS_DIR, "split_manifest_*.csv")):
    hits = glob.glob(os.path.join(SPLITS_DIR, "**", "split_manifest_*.csv"), recursive=True)
    if hits: SPLITS_DIR = os.path.dirname(hits[0])
print("splits dir:", SPLITS_DIR)

meta_path = os.path.join(SPLITS_DIR, "unified_classes.json")
meta = json.load(open(meta_path))
ALL_CLASSES = list(meta["all_classes"])
TASK_ORDER  = list(meta["task_order"])
print("tasks:", TASK_ORDER)
print("unified classes:", len(ALL_CLASSES))

rows_out, perclass_rows = [], []
grand = Counter()
for t in TASK_ORDER:
    mp = os.path.join(SPLITS_DIR, f"split_manifest_{t}.csv")
    if not os.path.exists(mp):
        print(f"  [warn] missing {mp}"); continue
    split_counts, cls_counts = Counter(), {}
    with open(mp, newline="") as f:
        for r in csv.DictReader(f):
            s = r["split"]; li = int(r["label_idx"])
            split_counts[s] += 1
            cls_counts.setdefault(li, Counter())[s] += 1
    total = sum(split_counts.values())
    grand.update(split_counts)
    rows_out.append({
        "task": t, "train": split_counts["train"], "val": split_counts["val"],
        "test": split_counts["test"], "total": total,
        "classes_present": len(cls_counts),
        "table1_cell": f'{split_counts["train"]} / {split_counts["val"]} / {split_counts["test"]}'})
    for li in sorted(cls_counts):
        c = cls_counts[li]
        perclass_rows.append({"task": t, "idx": li, "class": ALL_CLASSES[li],
                              "train": c["train"], "val": c["val"], "test": c["test"],
                              "total": sum(c.values())})

print("\n=== TABLE 1 — paste the last column into the Methods table ===")
print(f"{'Task':<10}{'Train':>8}{'Val':>7}{'Test':>7}{'Total':>8}{'Cls':>5}   Train / Val / Test")
for r in rows_out:
    print(f"{r['task']:<10}{r['train']:>8}{r['val']:>7}{r['test']:>7}{r['total']:>8}"
          f"{r['classes_present']:>5}   {r['table1_cell']}")
print(f"{'ALL':<10}{grand['train']:>8}{grand['val']:>7}{grand['test']:>7}"
      f"{sum(grand.values()):>8}")

# de-duplication report, if Phase 1 wrote one
ddp = glob.glob(os.path.join(SPLITS_DIR, "**", "dedup*.csv"), recursive=True)
if ddp:
    print(f"\n=== DE-DUPLICATION REPORT ({os.path.basename(ddp[0])}) ===")
    with open(ddp[0], newline="") as f:
        for i, line in enumerate(f):
            print("   ", line.rstrip())
            if i > 25: print("    ..."); break
    print("Report the total removed count in Section 2.3.")
else:
    print("\n[note] no dedup report found in the splits dataset. If Phase 1 wrote one "
          "elsewhere, quote its removal count in Section 2.3; otherwise state the protocol "
          "and that the removal count is available on request.")

try:
    import pandas as pd
    pd.DataFrame(rows_out).to_csv("table1_split_counts.csv", index=False)
    pd.DataFrame(perclass_rows).to_csv("tableS2_per_class_counts.csv", index=False)
    print("\nwrote table1_split_counts.csv and tableS2_per_class_counts.csv")
except Exception as e:
    print("[warn] csv write skipped:", e)

print("\n=== Supplementary Table S2 — per-class distribution (head) ===")
for r in perclass_rows[:12]:
    print(f"  {r['task']:<10}{r['idx']:>3}  {r['class']:<28}"
          f"{r['train']:>6}{r['val']:>5}{r['test']:>5}")
print(f"  ... {len(perclass_rows)} rows total")
