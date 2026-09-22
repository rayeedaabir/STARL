# Notebooks & data provenance

Every experiment in STARL was run as a Kaggle notebook, with intermediate outputs passed between stages as Kaggle datasets. This page lists all of them for full provenance. The four **source** image datasets (APTOS 2019, ODIR-5K, LAG, HAM10000) and the two external test sets are obtained from their original providers — see [Data availability](../README.md#data-availability).

## Notebooks

### 1 · Data preparation
- [STARL: Preprocess and Split](https://www.kaggle.com/code/rayeedaabir/starl-preprocess-and-split) — perceptual-hash de-duplication + frozen, class-stratified 70/15/15 split manifests
- [STARL: Table-1 Split Counts](https://www.kaggle.com/code/rejminislamnsu/starl-table-1-split-counts) — per-task / per-class counts (Table 1, Supplementary Table S2)

### 2 · Per-task training (main four-task sequence)
- [STARL: T1 APTOS Runner](https://www.kaggle.com/code/rayeedaabir/starl-t1-aptos-runner)
- [STARL: T2 ODIR Runner](https://www.kaggle.com/code/rayeedaabir/starl-t2-odir-runner)
- [STARL: T3 LAG Runner](https://www.kaggle.com/code/rayeedaabir/starl-t3-lag-runner)
- [STARL: T4 HAM10000 Runner v1](https://www.kaggle.com/code/rayeedaabir/starl-t4-ham10000-runner-v1) · [v2](https://www.kaggle.com/code/rayeedaabir/starl-t4-ham10000-runner-v2)

### 3 · KAN architecture runs
- [KAN Runner (T1)](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t1)
- KAN Runner (T2): [v1](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t2-v1) · [v2](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t2-v2)
- KAN Runner (T3): [v1](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t3-v1) · [v2](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t3-v2)
- KAN Runner (T4): [v1](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t4-v1) · [v2](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t4-v2) · [v3](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t4-v3) · [v4](https://www.kaggle.com/code/rayeedaabir/starl-kan-runner-t4-v4) · [v5](https://www.kaggle.com/code/rejminislamnsu/starl-kan-runner-t4-v5) · [v6](https://www.kaggle.com/code/rejminislamnsu/starl-kan-runner-t4-v6)

### 4 · Strategies & ablations
- [STARL: Joint run (4-model)](https://www.kaggle.com/code/rejminislamnsu/starl-joint-run-4-model) — joint (offline) upper bound
- Knowledge Distillation: [ResNet-50](https://www.kaggle.com/code/rejminislamnsu/starl-knowledge-distillation-resnet-50) · [ResNet-18](https://www.kaggle.com/code/rejminislamnsu/starl-knowledge-distillation-resnet-18)
- [STARL: Exemplar-Budget Ablation](https://www.kaggle.com/code/rayeedaabir/starl-exemplar-budget-ablation-old)
- [STARL: Task-Order Ablation](https://www.kaggle.com/code/rayeedaabir/starl-task-order-ablation)
- Session-wise Incremental Ablation: [v1](https://www.kaggle.com/code/rayeedaabir/starl-session-wise-incremental-ablation-v1) · [v2](https://www.kaggle.com/code/rejminislamnsu/starl-session-wise-incremental-ablation-v2)
- Seed-variance runners: [seed 2025 v1](https://www.kaggle.com/code/rejminislamnsu/starl-seed-2025-variance-runner-v1) · [seed 123 v1](https://www.kaggle.com/code/rayeedaabir/starl-seed-123-variance-runner-v1) · [seed 123 v2](https://www.kaggle.com/code/rayeedaabir/starl-seed-123-variance-runner-v2)
- [STARL: Baseline calculations](https://www.kaggle.com/code/rayeedaabir/starl-baseline-calculations) — majority-class baselines

### 5 · External validation
- [STARL: External Validation Study](https://www.kaggle.com/code/rejminislamnsu/starl-external-validation-study) — Messidor-2, Derm7pt
- [STARL: Base-Expert External Check](https://www.kaggle.com/code/rejminislamnsu/starl-base-expert-external-check)

### 6 · Analysis, diagnostics & reporting
- [STARL: CKA Activation-Path Study](https://www.kaggle.com/code/rejminislamnsu/starl-cka-activation-path-study)
- [STARL: Classifier-Bias Diagnosis + Post-hoc Correction](https://www.kaggle.com/code/rejminislamnsu/starl-classifier-bias-diagnosis-post-hoc-corr)
- [STARL: Significance + Bootstrap CI Tests](https://www.kaggle.com/code/rejminislamnsu/starl-significance-bootstrap-ci-tests)
- [STARL: Per-Image Predictions](https://www.kaggle.com/code/rejminislamnsu/starl-per-image-predictions)
- [STARL: Aggregate outputs](https://www.kaggle.com/code/rayeedaabir/starl-aggregate-outputs)
- [STARL: Final Report](https://www.kaggle.com/code/rayeedaabir/starl-final-report)

## Datasets (Kaggle-hosted)

### Core reproducibility
- [starl-splits](https://www.kaggle.com/datasets/rayeedaabir/starl-splits) — frozen split manifests, `unified_classes.json`, de-duplication report
- [starl-code](https://www.kaggle.com/datasets/rayeedaabir/starl-code) — analysis code snapshot

### Source image datasets (author-hosted mirrors)
- [Messidor-2 dataset](https://www.kaggle.com/datasets/rayeedaabir/messidor-2-dataset) — external DR test set
- [Derm7pt dataset](https://www.kaggle.com/datasets/rejminislamnsu/derm7pt-dataset) — external dermoscopy test set

*(APTOS 2019, ODIR-5K, LAG and HAM10000 come from their original providers — see the README's Data availability.)*

### Run-output datasets (intermediates passed between notebooks)
- Task outputs: [T1](https://www.kaggle.com/datasets/rayeedaabir/starl-t1-outputs) · [T2](https://www.kaggle.com/datasets/rayeedaabir/starl-t2-outputs) · [T3](https://www.kaggle.com/datasets/rayeedaabir/starl-t3-outputs) · [T4](https://www.kaggle.com/datasets/rayeedaabir/starl-t4-outputs)
- KAN outputs: [T1](https://www.kaggle.com/datasets/rayeedaabir/starl-t1-kan-outputs) · [T2](https://www.kaggle.com/datasets/rayeedaabir/starl-t2-kan-outputs) · [T3](https://www.kaggle.com/datasets/rayeedaabir/starl-t3-kan-outputs) · [T4](https://www.kaggle.com/datasets/rayeedaabir/starl-t4-kan-outputs)
- Seed variance: [seed123-outputs](https://www.kaggle.com/datasets/rayeedaabir/seed123-outputs) · [seed2025-outputs](https://www.kaggle.com/datasets/rejminislamnsu/seed2025-outputs)
- [STARL-4tasks-12models-6methods](https://www.kaggle.com/datasets/rayeedaabir/starl-4tasks-12models-6methods) — consolidated main-run outputs
- [STARL-T4-ResNet50-path](https://www.kaggle.com/datasets/rayeedaabir/starl-t4-resnet50-path) — ResNet-50 path for distillation
- [STARL-Aggregate-outputs-final](https://www.kaggle.com/datasets/rayeedaabir/starl-aggregate-outputs-final) — final aggregated result tables
- [STARL-joined-outputs](https://www.kaggle.com/datasets/rayeedaabir/starl-joined-outputs) — joined outputs for per-image prediction / CI analysis
