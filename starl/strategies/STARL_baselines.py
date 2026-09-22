# =============================================================================
# STARL — majority-class baselines per task, per split
# =============================================================================
# WHY: Section 3.3 compares the accuracy retained on each preceding task. Those
# tasks differ in class count (APTOS 5, ODIR 7, LAG 2), so their majority-class
# baselines differ too, and a raw retention figure is not comparable across them.
# This computes the exact baseline for every task's FROZEN TEST partition so the
# retention numbers can be reported as skill above baseline.
#
# ATTACH: the `starl-splits` dataset ONLY. No GPU, no torch, runs in seconds.
# Paste into a Kaggle cell and run, or from a shell:
#     python STARL_baselines.py [/path/to/starl-splits]
#
# Manifest schema (from starl_core.read_manifest): columns `filepath`,
# `label_idx`, `split`; files named split_manifest_<TASK>.csv.
# =============================================================================
import sys, os, json, collections, csv

# ---- locate the splits directory -------------------------------------------
# NOTE: in a notebook sys.argv[1] is the kernel's own argument, not a path, so
# a bare "len(sys.argv) > 1" test picks up garbage. Only accept an argument that
# is actually an existing directory.
CANDIDATES = []
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    CANDIDATES.append(sys.argv[1])
CANDIDATES += ["/kaggle/input/starl-splits", "./starl-splits", "."]

def has_manifests(d):
    if not os.path.isdir(d):
        return False
    for _r, _dirs, files in os.walk(d):
        if any(f.startswith("split_manifest") and f.endswith(".csv") for f in files):
            return True
    return False

SPLITS = next((c for c in CANDIDATES if has_manifests(c)), None)

if SPLITS is None:
    # Last resort: sweep everything mounted under /kaggle/input.
    if os.path.isdir("/kaggle/input"):
        for entry in sorted(os.listdir("/kaggle/input")):
            p = os.path.join("/kaggle/input", entry)
            if has_manifests(p):
                SPLITS = p
                break

if SPLITS is None:
    print("ERROR: no split_manifest_*.csv found. Looked in:")
    for c in CANDIDATES:
        print(f"   {c}   (exists: {os.path.isdir(c)})")
    if os.path.isdir("/kaggle/input"):
        print("\n/kaggle/input contains:")
        for entry in sorted(os.listdir("/kaggle/input")):
            print("   ", entry)
        print("\nAttach the `starl-splits` dataset, or set SPLITS by hand below.")
    raise SystemExit(1)

print("using splits directory:", SPLITS)

# ---- class names (optional, for readable output) ----------------------------
ALL = None
for _r, _dirs, files in os.walk(SPLITS):
    if "unified_classes.json" in files:
        j = json.load(open(os.path.join(_r, "unified_classes.json")))
        ALL = j.get("all_classes", j) if isinstance(j, dict) else j
        break

# ---- read every manifest ----------------------------------------------------
manifests = []
for _r, _dirs, files in os.walk(SPLITS):
    for f in files:
        if f.startswith("split_manifest") and f.endswith(".csv"):
            manifests.append(os.path.join(_r, f))
manifests.sort()
print(f"found {len(manifests)} manifest(s)")

LABEL_COLS = ("label_idx", "label", "class_idx", "class", "y")   # label_idx is the real one
SPLIT_COLS = ("split", "partition", "subset")
VAL_ALIASES = {"val": "val", "valid": "val", "validation": "val"}

rows = []
for m in manifests:
    task = os.path.basename(m).replace("split_manifest_", "").replace(".csv", "")
    per_split = collections.defaultdict(collections.Counter)
    with open(m, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
        scol = next((cols[c] for c in SPLIT_COLS if c in cols), None)
        lcol = next((cols[c] for c in LABEL_COLS if c in cols), None)
        if not scol or not lcol:
            print(f"  [skip] {os.path.basename(m)}: columns are {rd.fieldnames}")
            continue
        for r in rd:
            s = r[scol].strip().lower()
            per_split[VAL_ALIASES.get(s, s)][r[lcol]] += 1

    for split in ("train", "val", "test"):
        c = per_split.get(split)
        if not c:
            continue
        n = sum(c.values())
        top, topn = c.most_common(1)[0]
        name = ALL[int(top)] if (ALL and str(top).lstrip("-").isdigit() and int(top) < len(ALL)) else str(top)
        rows.append(dict(task=task, split=split, n_images=n, n_classes=len(c),
                         majority_class=name, majority_n=topn,
                         majority_baseline=round(topn / n, 4),
                         chance=round(1.0 / len(c), 4)))

if not rows:
    raise SystemExit("ERROR: manifests were found but none had usable split/label columns — "
                     "see the [skip] lines above for the column names actually present.")

print(f"\n{'task':<14}{'split':<7}{'images':>8}{'cls':>5}  {'majority class':<24}{'baseline':>10}{'chance':>8}")
for r in rows:
    print(f"{r['task']:<14}{r['split']:<7}{r['n_images']:>8}{r['n_classes']:>5}  "
          f"{str(r['majority_class'])[:23]:<24}{r['majority_baseline']:>10.4f}{r['chance']:>8.4f}")

out = "/kaggle/working/majority_baselines.csv" if os.path.isdir("/kaggle/working") else "majority_baselines.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nwrote", out)

# ---- the numbers Section 3.3 needs ------------------------------------------
test = {r["task"]: r for r in rows if r["split"] == "test"}
print("\nFOR TABLE 9 — test-partition majority baselines:")
for t in sorted(test):
    r = test[t]
    print(f"  {t:<14} {r['majority_baseline']:.4f}   ({r['majority_class']}, "
          f"{r['majority_n']}/{r['n_images']}, {r['n_classes']} classes)")

apt = next((r for t, r in test.items() if "APTOS" in t.upper()), None)
if apt:
    ok = abs(apt["majority_baseline"] - 0.4664) < 0.0006
    print(f"\nSANITY CHECK  APTOS test baseline = {apt['majority_baseline']:.4f} "
          f"(expected 0.4664)  ->  {'PASS' if ok else '*** FAIL — wrong manifest, do not use these numbers ***'}")
else:
    print("\nSANITY CHECK could not run: no APTOS task found among", sorted(test))
