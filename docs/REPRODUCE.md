# Reproducing STARL

This guide maps the pipeline to the scripts in `starl/`. The design principle throughout: **every image is read through a frozen split manifest**, so no image can leak across train/val/test or across tasks.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

GPU recommended for training; Phase 1 and most analysis run on CPU.

## 1. Datasets

Download the four datasets from their original providers (see the README's *Data availability*) and note their local paths. STARL pools every labelled image and re-splits it under one protocol, so any provider layout works.

## 2. Phase 1 — de-duplication + frozen splits

```bash
python starl/preprocess_and_split.py     # (01_preprocess_and_split.py)
```

Point the dataset paths at the top of the script first. It:

- computes a 64-bit perceptual hash for every image and removes near-duplicates in task order (earliest task wins),
- writes a **frozen, class-stratified 70/15/15** split per task, and
- emits `manifests/` — `split_manifest_<task>.csv`, `unified_classes.json`, and `dedup_report.csv`.

Downstream scripts depend only on these manifests. To reproduce the exact partitions used in the paper, use the deposited `manifests/` from the Zenodo record instead of re-running this step.

## 3. Phase 2 — training and strategies

Per-task training entry points and the continual-learning strategies:

| Step | Script | Produces |
|---|---|---|
| Base + task runners | `runners/T1_APTOS_runner.py`, `runners/T2_ODIR_runner.py`, … | per-task checkpoints + metrics |
| Lower/upper bounds & baselines | `strategies/STARL_baselines.py`, `strategies/STARL_joint.py` | naive, joint (offline) reference |
| Rehearsal / exemplars | `strategies/STARL_exemplar.py` | fixed per-class budgets (m = 1, 5, 20), herding vs random |
| Knowledge distillation | `strategies/STARL_kd.py` | MobileNetV2 student |
| Classifier-bias correction | `strategies/STARL_biascorrect.py` | post-hoc final-layer test |
| Multi-seed sweep | `strategies/STARL_multiseed.py` | seeds 42, 123, 2025 |
| Session-wise protocol | `strategies/STARL_sessions.py` | 16 single-class sessions |
| Task reordering | `strategies/STARL_taskorder.py` | dissimilar-task-first permutation |
| External validation | `strategies/STARL_external.py`, `strategies/STARL_external_base.py` | Messidor-2, Derm7pt |
| KAN model | `runners/KAN_runner.py` (`KAN_resume_check.py`) | KAN architecture runs |

## 4. Phase 3 — analysis, tables, figures

| Output | Script |
|---|---|
| Aggregate metrics / result tables | `analysis/STARL_aggregate.py`, `analysis/STARL_final_report.py` |
| Representation similarity (CKA) | `analysis/STARL_cka.py` |
| Significance + bootstrap CIs | `analysis/STARL_significance.py` |
| Per-image predictions (→ per-class recall) | `analysis/STARL_predictions.py` |
| Table 1 / Supplementary S2 counts | `analysis/STARL_table1_counts.py` |

## Notes

- The scripts share one import, `starl_core`. If you keep them in subfolders, add an `__init__.py` and import as `from starl.starl_core import …`; otherwise keep them flat under `starl/`.
- Torch, `timm`, and the KAN package are imported lazily, so the pure-Python utilities (metrics, split counts, manifest reading) run without a GPU stack installed.
