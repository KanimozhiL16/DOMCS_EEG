# DOMCS-EEG: Disentangled Orthogonal Multi-Constraint State-Invariant EEG Biometric Verification

**IEEE Transactions on Information Forensics and Security (T-IFS) — T-IFS-26761-2026**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

---

## Overview

DOMCS-EEG learns **state-invariant** EEG biometric representations by simultaneously optimising four constraints:

| Loss | Symbol | Purpose |
|------|--------|---------|
| ArcFace (identity) | L_arc | Discriminative identity embeddings z_id |
| Supervised Contrastive | L_supcon | Within-class compactness of z_id |
| State Classification | L_state | Forces z_state to encode cognitive state |
| Orthogonality | L_orth | Pushes z_id ⊥ z_state in embedding space |

The **f.detach()** barrier in the state branch ensures state and orthogonality losses never propagate into the shared encoder through the state path. Identity-label supervision is applied to REST windows only (B2T-safe training).

**Key result — PHYSIONET B2T Protocol:**
- EER = **2.4138% ± 0.0906%** (5 seeds, 109 subjects)
- AUC = **0.9962 ± 0.0004**
- Bootstrap 95% CI: [2.3354%, 2.4796%]

---

## Architecture

```
Input (B, 64, 256) — 64 EEG channels, 256 samples (2s @ 128 Hz)
  │
  └─ EEGEncoder [Conv k=7 → Conv k=5 → Conv k=3 | BN | ELU | AdaptiveAvgPool]
       │                           → f ∈ ℝ²⁵⁶
       ├─ IdentityBranch [Linear(256→128) → LayerNorm → L2-norm]
       │                           → z_id ∈ ℝ¹²⁸   (grad flows to encoder)
       └─ StateBranch    [f.detach() → Linear(256→128) → LayerNorm → L2-norm]
                                   → z_state ∈ ℝ¹²⁸ (detach: no grad to encoder)
```

Parameters: 234,880 total (verified).

---

## Repository Structure

```
DOMCS-EEG/
├── main/                    ← Core DOMCS-EEG implementation (PHYSIONET)
│   ├── config.py            ← Locked hyperparameters (single source of truth)
│   ├── model.py             ← EEGEncoder + IdentityBranch + StateBranch + ArcFaceHead
│   ├── losses.py            ← ArcFace, SupCon, State CE, Orthogonality losses
│   ├── data_loader.py       ← B2T-safe data loading and splitting
│   ├── 01_train_dual_space_domcs.py   ← Main training script (5 seeds)
│   ├── 02_evaluate_b2t.py             ← B2T verification evaluation
│   ├── 03_generate_figures.py         ← Training curve and ROC figures
│   ├── 04_security_analysis.py        ← T0–T5 threat model evaluation
│   ├── 05_disentanglement_analysis.py ← Orthogonality and UMAP analysis
│   ├── 06_generate_tables.py          ← Supplementary tables (CSV + LaTeX)
│   ├── 07_export_ieee_figures.py      ← IEEE-format figure export
│   ├── 08_deep_analysis.py            ← Per-task EER and similarity analysis
│   ├── 09_ablation_study.py           ← Ablation E1–E5 (5 seeds each)
│   └── 10_protocol_comparison.py      ← B2T vs P2/P3 protocol comparison
├── cross_dataset/
│   ├── bed_domcs_locked.py  ← BED cross-dataset validation (14ch, BED-B2T)
│   └── bed_baselines.py     ← BED baselines: EEGNet, DeepConvNet, CNN+ArcFace
├── baselines/
│   └── dann_physionet.py    ← DANN baseline (PHYSIONET, Table S8)
├── visualization/
│   ├── bed_visualize.py     ← BED dataset figures (6 figures)
│   └── bed_disentangle_viz.py ← BED disentanglement UMAP + orthogonality
├── data/
│   └── README.md            ← Data access instructions
├── requirements.txt
├── RESULTS.md               ← All verified numerical results
└── LICENSE
```

---

## Datasets

### PHYSIONET EEGMMIDB (primary)
- 109 subjects, 14 EEG motor imagery runs (R01–R14), 64 channels, 160 Hz → resampled to 128 Hz
- Windowed: 2s, 1s step → 173,198 windows (X: N×64×256)
- **Download:** https://physionet.org/content/eegmmidb/1.0.0/
- **Preprocessing:** Use `mne` to load `.edf` files; window and save as `.npz` with keys `X`, `y`, `session`

### BED Dataset (cross-dataset validation)
- 21 subjects, 3 sessions (r01 REST, r02 REST, r03 TASK), 14 channels (Emotiv EPOC+), 128 Hz
- Windowed: 2s, 1s step
- **Access:** Contact the BED dataset authors for access

---

## B2T Protocol (Brain-to-Task)

| Split | Sessions | Purpose |
|-------|----------|---------|
| Enrollment | R01 + R02 (REST) | Build KMeans gallery (K=3 prototypes/subject) |
| Verification | R03–R14 (TASK) | Probe windows — cognitively active state |

This is the **only scientifically valid** protocol for real-world deployment: biometric templates captured at rest; verification during active cognitive use. Protocols that mix enrollment and probe states (P2, P3) report artificially optimistic EER and are included only for comparison.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/DOMCS-EEG.git
cd DOMCS-EEG
pip install -r requirements.txt
```

**Hardware:** NVIDIA GPU with ≥8 GB VRAM recommended. All results were produced on an NVIDIA A100 (Brev.dev cloud instance).

---

## Usage

### 1. Prepare data

Place the preprocessed PHYSIONET NPZ at `data/preprocessed_b2t.npz` with keys:
- `X`: shape (N, 64, 256), float32
- `y`: shape (N,), int64 — subject index (0–108)
- `session`: shape (N,), str — run label ('R01', 'R02', ..., 'R14')

### 2. Train (5 seeds)

```bash
cd main/
python 01_train_dual_space_domcs.py
```

Results saved to `./DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/` (configurable in `config.py`).

### 3. Evaluate B2T

```bash
python 02_evaluate_b2t.py
```

### 4. Cross-dataset (BED)

```bash
python cross_dataset/bed_domcs_locked.py \
    --data /path/to/BED_win2s_step1s_fs128.npz \
    --out ./BED_DOMCS_RESULTS

python cross_dataset/bed_baselines.py \
    --data /path/to/BED_win2s_step1s_fs128.npz \
    --out ./BED_BASELINE_RESULTS
```

### 5. DANN baseline (Table S8)

```bash
python baselines/dann_physionet.py \
    --data /path/to/preprocessed_b2t.npz \
    --out ./DANN_RESULTS
```

### 6. Visualization

```bash
python visualization/bed_visualize.py \
    --data /path/to/BED_win2s_step1s_fs128.npz \
    --out ./BED_VIZ

python visualization/bed_disentangle_viz.py \
    --data /path/to/BED_win2s_step1s_fs128.npz \
    --ckpt-dir ./BED_DOMCS_RESULTS \
    --out ./BED_VIZ
```

---

## Locked Hyperparameters

All results in the paper were produced with these exact values (see `main/config.py`):

| Parameter | Value | Note |
|-----------|-------|------|
| ARC_S | 32.0 | ArcFace scale |
| ARC_M | 0.50 | ArcFace margin |
| LAMBDA_SUPCON | 0.30 | SupCon loss weight |
| LAMBDA_STATE | 0.50 | State CE loss weight |
| LAMBDA_ORTH | 0.10 | Orthogonality loss weight |
| VAL_SPLIT | 0.10 | Validation fraction |
| EPOCHS | 60 | Training epochs |
| BATCH_SIZE | 128 | Batch size |
| LR | 3e-4 | Adam learning rate |
| SEEDS | [1,2,3,4,5] | Reproducibility seeds |

---

## Reproducibility

All reported numbers are averages over 5 independent random seeds. The `SEEDS = [1,2,3,4,5]` list in `config.py` is fixed and must not be changed to reproduce paper results.

Training is deterministic given the same seed, hardware, and PyTorch/CUDA versions. Minor numerical differences (<0.01% EER) may occur across different GPU models due to floating-point non-determinism in cuDNN operations.

**Verified on:** NVIDIA A100 (40 GB), Python 3.10, PyTorch 2.1.0, CUDA 12.1.

---

## Citation

If you use this code, please cite:

```bibtex
@article{kanimozhi2026domcseeg,
  title     = {DOMCS-EEG: Disentangled Orthogonal Multi-Constraint State-Invariant
               EEG Biometric Verification},
  author    = {Kanimozhi, L. and {co-authors}},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2026},
  note      = {Under review, manuscript T-IFS-26761-2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

This work uses the PHYSIONET EEG Motor Movement/Imagery Dataset (Goldberger et al., PhysioBank, 2000) and the BED dataset. Experiments were conducted on Brev.dev cloud GPU infrastructure.
