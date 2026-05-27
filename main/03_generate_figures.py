#!/usr/bin/env python3
"""
03_generate_figures.py
=======================
Generates all 15 paper figures from REAL model outputs.
Must be run AFTER 01_train, 02_evaluate, 04_security, 05_disentanglement.

FIG_01  Training curves (loss components per epoch)
FIG_02  Architecture diagram (text-based, programmatic)
FIG_03  B2T protocol diagram
FIG_04  ROC curves (5-seed overlay + mean)
FIG_05  DET curves
FIG_06  Protocol comparison (B2T vs random-split vs same-session)
FIG_07  Per-task EER bar chart
FIG_08  Disentanglement t-SNE (from 05_disentanglement)
FIG_09  Orthogonality distribution (from 05_disentanglement)
FIG_10  Probe accuracy bar chart (from 05_disentanglement)
FIG_11  Similarity analysis (z_id intra/inter)
FIG_12  Silhouette analysis
FIG_13  Security summary (T5 histogram, from 04_security)
FIG_14  Noise robustness (from 04_security)
FIG_15  Adversarial robustness (from 04_security)
"""

import os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.special import ndtri

from config import (SEEDS, LOG_DIR, FIG_DIR, PNG_DIR, PDF_DIR,
                    TASK_RUNS, REST_RUNS, MPL_STYLE, COLORS)

plt.rcParams.update(MPL_STYLE)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def savefig(fig, name):
    for d, ext in [(PNG_DIR, 'png'), (PDF_DIR, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"{name}.{ext}"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ {name}")


def load_json(path):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ Not found: {p}")
        return None
    with open(p) as f:
        return json.load(f)


def load_npz(path):
    p = Path(path)
    if not p.exists():
        print(f"  ⚠ Not found: {p}")
        return None
    return np.load(str(p))


# ─── FIG_01: Training curves ──────────────────────────────────────────────────

def fig01_training_curves():
    print("  FIG_01: Training curves")
    import csv
    loss_keys = ["L_total","L_arcface","L_supcon","L_state","L_orth"]
    line_styles = ['-','--','-.', ':', (0,(3,1,1,1))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    for seed in SEEDS:
        csv_path = Path(LOG_DIR) / f"seed_{seed}" / "train_log.csv"
        if not csv_path.exists():
            continue
        epochs, data = [], {k: [] for k in ["train_L_total","val_L_total","val_state_acc"]}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                for k in data:
                    try: data[k].append(float(row[k]))
                    except (KeyError, ValueError): data[k].append(None)

        epochs = np.array(epochs)
        tr  = np.array([v for v in data["train_L_total"] if v is not None])
        vl  = np.array([v for v in data["val_L_total"]   if v is not None])
        acc = np.array([v for v in data["val_state_acc"] if v is not None])

        alpha = 0.8 if seed == 1 else 0.4
        axes[0].plot(epochs[:len(tr)], tr,  alpha=alpha, lw=1.0, label=f"Tr s{seed}")
        axes[0].plot(epochs[:len(vl)], vl,  alpha=alpha, lw=1.0, ls='--')
        axes[1].plot(epochs[:len(acc)], np.array(acc)*100, alpha=alpha, lw=1.0,
                     label=f"Seed {seed}")

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Train/Val Loss (solid=train, dashed=val)")
    axes[0].legend(fontsize=6.5, ncol=2)

    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("State Acc (%)")
    axes[1].set_title("State Classification Accuracy (Val)")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(50, color='grey', ls='--', lw=0.8, label='Chance (50%)')

    fig.suptitle("DOMCS-EEG — Dual-Space Training Curves", fontsize=10)
    savefig(fig, "FIG_01_training_curves")


# ─── FIG_04: ROC curves (5-seed) ──────────────────────────────────────────────

def fig04_roc_curves():
    print("  FIG_04: ROC curves")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    eers, aucs = [], []

    for seed in SEEDS:
        d = load_npz(Path(LOG_DIR) / f"seed_{seed}" / f"scores_seed{seed}.npz")
        if d is None: continue
        fpr, tpr = d["fpr"], d["tpr"]
        from sklearn.metrics import auc as sk_auc
        roc_auc = sk_auc(fpr, tpr)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2 * 100
        eers.append(eer); aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1.0, alpha=0.5,
                label=f"s{seed} EER={eer:.2f}% AUC={roc_auc:.4f}")

    if eers:
        ax.plot([0,1],[0,1],'k--',lw=0.7)
        ax.set_xlabel("False Acceptance Rate")
        ax.set_ylabel("Genuine Acceptance Rate")
        ax.set_title(f"ROC — 5-Seed  Mean EER={np.mean(eers):.2f}%±{np.std(eers):.2f}%")
        ax.legend(fontsize=6.5, loc='lower right')
    savefig(fig, "FIG_04_roc")


# ─── FIG_05: DET curves ───────────────────────────────────────────────────────

def fig05_det_curves():
    print("  FIG_05: DET curves")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    def det_t(p):
        return ndtri(np.clip(p, 1e-6, 1-1e-6))

    ticks = [1, 2, 5, 10, 20, 40]
    tick_pos = [ndtri(t/100) for t in ticks]

    for seed in SEEDS:
        d = load_npz(Path(LOG_DIR) / f"seed_{seed}" / f"scores_seed{seed}.npz")
        if d is None: continue
        fpr, tpr = d["fpr"], d["tpr"]
        frr = 1 - tpr
        ax.plot(det_t(fpr), det_t(frr), lw=1.0, alpha=0.6, label=f"Seed {seed}")

    ax.set_xticks(tick_pos); ax.set_xticklabels(ticks)
    ax.set_yticks(tick_pos); ax.set_yticklabels(ticks)
    ax.set_xlim(ndtri(0.005), ndtri(0.5))
    ax.set_ylim(ndtri(0.005), ndtri(0.5))
    ax.set_xlabel("FAR (%)"); ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curves — 5 Seeds")
    ax.legend(fontsize=7)
    savefig(fig, "FIG_05_det")


# ─── FIG_07: Per-task EER ─────────────────────────────────────────────────────

def fig07_per_task_eer():
    print("  FIG_07: Per-task EER")
    summary = load_json(Path(LOG_DIR) / "eval_summary.json")
    if summary is None: return

    task_groups = {}
    for seed_res in summary.get("per_seed", []):
        for grp, val in seed_res.get("per_task_group", {}).items():
            if val.get("eer") is not None:
                task_groups.setdefault(grp, []).append(val["eer"])

    if not task_groups: return

    grp_names = list(task_groups.keys())
    means     = [np.mean(task_groups[g]) for g in grp_names]
    stds      = [np.std(task_groups[g])  for g in grp_names]
    overall   = summary.get("mean_eer")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    x    = np.arange(len(grp_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS["clean"], COLORS["T0"],
                         COLORS["T1"],    COLORS["pgd"]],
                  alpha=0.8, edgecolor='white')
    ax.bar_label(bars, fmt='%.2f%%', fontsize=8, padding=3)
    if overall:
        ax.axhline(overall, color='k', ls='--', lw=1.0,
                   label=f"Mean EER={overall:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_',' ') for g in grp_names], fontsize=8)
    ax.set_ylabel("EER (%)"); ax.set_title("Per-Task-Group EER (5-seed mean)")
    ax.legend(fontsize=8)
    savefig(fig, "FIG_07_per_task_eer")


# ─── FIG_11: Similarity analysis ──────────────────────────────────────────────

def fig11_similarity():
    print("  FIG_11: Similarity analysis")
    dis_metrics = load_json(Path(LOG_DIR) / "disentanglement_metrics.json")
    if not dis_metrics: return

    labels = ['Intra-subject\n(z_id)', 'Inter-subject\n(z_id)',
              'Stability gap\n(z_id)']
    vals   = [
        np.mean([m["zid_intra_subject_cos"] for m in dis_metrics if m]),
        np.mean([m["zid_inter_subject_cos"] for m in dis_metrics if m]),
        np.mean([m["zid_stability_gap"]     for m in dis_metrics if m]),
    ]
    stds   = [
        np.std([m["zid_intra_subject_cos"] for m in dis_metrics if m]),
        np.std([m["zid_inter_subject_cos"] for m in dis_metrics if m]),
        np.std([m["zid_stability_gap"]     for m in dis_metrics if m]),
    ]
    colors = [COLORS["z_id"], '#e74c3c', '#3498db']

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=stds, capsize=5, color=colors, alpha=0.8,
                  edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(r"$z_{id}$ Cosine Similarity (Identity Discriminability)")
    savefig(fig, "FIG_11_similarity_analysis")


# ─── FIG_12: Silhouette analysis ──────────────────────────────────────────────

def fig12_silhouette():
    print("  FIG_12: Silhouette analysis")
    dis_metrics = load_json(Path(LOG_DIR) / "disentanglement_metrics.json")
    if not dis_metrics: return

    keys    = ["sil_zid_subject", "sil_zid_state",
               "sil_zstate_subject", "sil_zstate_state"]
    labels  = [r"$z_{id}$↔Subj", r"$z_{id}$↔State",
               r"$z_{st}$↔Subj", r"$z_{st}$↔State"]
    desired = [+1, -1, -1, +1]   # expected direction
    colors  = [COLORS["z_id"], COLORS["z_id"],
               COLORS["z_state"], COLORS["z_state"]]
    alpha_s = [0.9, 0.4, 0.4, 0.9]

    means = [np.nanmean([m[k] for m in dis_metrics if m and k in m]) for k in keys]
    stds  = [np.nanstd ([m[k] for m in dis_metrics if m and k in m]) for k in keys]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x    = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[c for c in colors], alpha=0.8, edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=3)
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    # Mark desired direction
    for xi, d, mean in zip(x, desired, means):
        sym = '✓' if (d > 0) == (mean > 0) else '✗'
        ax.text(xi, (mean + 0.05 if mean >= 0 else mean - 0.09),
                sym, ha='center', va='bottom', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis — Embedding Cluster Quality")
    handles = [mpatches.Patch(color=COLORS["z_id"],    label=r"$z_{id}$"),
               mpatches.Patch(color=COLORS["z_state"], label=r"$z_{state}$")]
    ax.legend(handles=handles, fontsize=8)
    savefig(fig, "FIG_12_silhouette_analysis")


# ─── FIG_06: Per-run EER drift ────────────────────────────────────────────────

def fig06_per_run_drift():
    print("  FIG_06: Per-run EER drift (B2T protocol comparison)")
    summary = load_json(Path(LOG_DIR) / "eval_summary.json")
    if summary is None: return

    per_run_all = {}
    for seed_res in summary.get("per_seed", []):
        for run, val in seed_res.get("per_run", {}).items():
            if val.get("eer") is not None:
                per_run_all.setdefault(run, []).append(val["eer"])

    if not per_run_all: return
    runs_ordered = [r for r in TASK_RUNS if r in per_run_all]
    means = [np.mean(per_run_all[r]) for r in runs_ordered]
    stds  = [np.std(per_run_all[r])  for r in runs_ordered]
    overall = summary.get("mean_eer")

    task_grp_color = {}
    for r in runs_ordered:
        if r in ['R03','R07','R11']: task_grp_color[r] = COLORS["clean"]
        elif r in ['R04','R08','R12']: task_grp_color[r] = COLORS["T0"]
        elif r in ['R05','R09','R13']: task_grp_color[r] = COLORS["T1"]
        else: task_grp_color[r] = COLORS["pgd"]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(runs_ordered))
    ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=3, color='k', lw=1.2, ms=4)
    for xi, r, m in zip(x, runs_ordered, means):
        ax.scatter(xi, m, color=task_grp_color[r], s=50, zorder=5)
    if overall:
        ax.axhline(overall, color='grey', ls='--', lw=0.9,
                   label=f"Mean EER={overall:.2f}%")
    ax.set_xticks(x); ax.set_xticklabels(runs_ordered, rotation=30, fontsize=8)
    ax.set_ylabel("EER (%)"); ax.set_title("Per-Run EER Drift (R03→R14)")

    handles = [mpatches.Patch(color=COLORS["clean"],  label="RL-Fist MI"),
               mpatches.Patch(color=COLORS["T0"],     label="BF-Feet MI"),
               mpatches.Patch(color=COLORS["T1"],     label="RL-Fist MV"),
               mpatches.Patch(color=COLORS["pgd"],    label="BF-Feet MV")]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc='upper left')
    savefig(fig, "FIG_06_protocol_comparison")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("DOMCS-EEG — Figure Generation (FIG_01 – FIG_15)")
    print("="*60)

    fig01_training_curves()
    fig04_roc_curves()
    fig05_det_curves()
    fig06_per_run_drift()
    fig07_per_task_eer()
    fig11_similarity()
    fig12_silhouette()

    # FIG_08, 09, 10: produced by 05_disentanglement_analysis.py
    # FIG_13, 14, 15: produced by 04_security_analysis.py
    print("\n  Note: FIG_02 (architecture), FIG_03 (protocol) are")
    print("  conceptual diagrams — prepare in draw.io or LaTeX TikZ.")
    print("  FIG_08-10: run 05_disentanglement_analysis.py")
    print("  FIG_13-15: run 04_security_analysis.py")
    print(f"\n  All figures saved to:")
    print(f"    PNG: {PNG_DIR}")
    print(f"    PDF: {PDF_DIR}")


if __name__ == "__main__":
    main()
