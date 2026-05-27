#!/usr/bin/env python3
"""
05_disentanglement_analysis.py
================================
Disentanglement verification for DOMCS-EEG.
All metrics computed from REAL model embeddings.

Outputs:
  - z_id probe accuracy (subject)
  - z_state probe accuracy (state)
  - Cross-probe: z_id → state leakage, z_state → subject leakage
  - Mutual information: MI(z_id, subject), MI(z_id, state), MI(z_state, subject), MI(z_state, state)
  - Orthogonality: mean |cosine(z_id, z_state)|
  - Silhouette scores: Sil(z_id, subject), Sil(z_id, state), Sil(z_state, subject), Sil(z_state, state)
  - t-SNE visualisations (FIG_08)
  - FIG_09: orthogonality distribution
  - FIG_10: probe accuracy bar chart
  - FIG_11: similarity analysis
  - FIG_12: silhouette analysis
"""

import os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (SEEDS, CKPT_DIR, LOG_DIR, FIG_DIR, PNG_DIR, PDF_DIR,
                    MPL_STYLE, COLORS)
from model       import DOMCSEEGModel
from data_loader import load_npz, prepare_training_data, DualSpaceDataset
from torch.utils.data import DataLoader
import torch

plt.rcParams.update(MPL_STYLE)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_embeddings_full(model, X, y_subj, y_state, device, batch_size=512):
    """Extract z_id and z_state for a full dataset."""
    model.eval()
    ds     = DualSpaceDataset(X, y_subj.astype(np.int64), y_state.astype(np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    all_zid, all_zst, all_ys, all_ystate = [], [], [], []
    with torch.no_grad():
        for xb, yb, sb in loader:
            xb = xb.to(device)
            z_id, z_state, _ = model(xb)
            all_zid.append(z_id.cpu().numpy())
            all_zst.append(z_state.cpu().numpy())
            all_ys.append(yb.numpy())
            all_ystate.append(sb.numpy())
    return (np.concatenate(all_zid),
            np.concatenate(all_zst),
            np.concatenate(all_ys),
            np.concatenate(all_ystate))


def linear_probe(embeddings, labels, n_splits=5, seed=42):
    """Train a linear probe classifier and return accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    le     = LabelEncoder()
    labels = le.fit_transform(labels)
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs   = []
    for tr, te in skf.split(embeddings, labels):
        clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
        clf.fit(embeddings[tr], labels[tr])
        accs.append(clf.score(embeddings[te], labels[te]))
    return float(np.mean(accs)), float(np.std(accs))


def compute_mutual_information(embeddings, labels, n_bins=20):
    """Estimate MI(embeddings, labels) via discretisation."""
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(embeddings, labels, random_state=42)
    return float(mi.mean())


def compute_silhouette(embeddings, labels, max_samples=5000, seed=42):
    """Silhouette score for cluster quality."""
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_samples:
        idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]
    try:
        return float(silhouette_score(embeddings, labels, metric='cosine',
                                      random_state=seed))
    except Exception:
        return float('nan')


def compute_orthogonality(z_id, z_state):
    """Mean |cosine(z_id, z_state)| — both already L2-normalised."""
    cosines = np.sum(z_id * z_state, axis=1)  # dot product = cosine
    return float(np.mean(np.abs(cosines))), float(np.std(np.abs(cosines))), cosines


def run_disentanglement(seed, X, y, session, device):
    print(f"\n  Disentanglement analysis — Seed {seed}")
    ckpt_path = Path(CKPT_DIR) / f"seed_{seed}" / "model_best.pt"
    if not ckpt_path.exists():
        print(f"  ⚠ Checkpoint not found — skip")
        return None

    model = DOMCSEEGModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Prepare full dataset (both rest + task)
    from data_loader import prepare_training_data
    _, val_ds, _ = prepare_training_data(X, y, session, val_split=0.20, seed=seed)
    # Use validation set for unbiased analysis
    X_val    = val_ds.X.numpy()
    ys_val   = val_ds.y_subj.numpy()
    ystate_val = val_ds.y_state.numpy()

    print(f"    Embedding {len(X_val):,} validation samples...")
    z_id, z_state, y_subj, y_state = extract_embeddings_full(
        model, X_val, ys_val, ystate_val, device)

    # ── Linear probes ─────────────────────────────────────────────
    print("    Running linear probes...")
    zid_subj_acc, zid_subj_std    = linear_probe(z_id,    y_subj)
    zid_state_acc, zid_state_std  = linear_probe(z_id,    y_state)
    zst_subj_acc, zst_subj_std    = linear_probe(z_state, y_subj)
    zst_state_acc, zst_state_std  = linear_probe(z_state, y_state)
    chance_subj  = 1.0 / len(np.unique(y_subj))
    chance_state = 1.0 / len(np.unique(y_state))

    # ── Mutual information ────────────────────────────────────────
    print("    Computing mutual information...")
    mi_zid_subj    = compute_mutual_information(z_id,    y_subj)
    mi_zid_state   = compute_mutual_information(z_id,    y_state)
    mi_zst_subj    = compute_mutual_information(z_state, y_subj)
    mi_zst_state   = compute_mutual_information(z_state, y_state)

    # ── Orthogonality ─────────────────────────────────────────────
    orth_mean, orth_std, cosines = compute_orthogonality(z_id, z_state)
    print(f"    Orthogonality: mean|cos|={orth_mean:.4f} ± {orth_std:.4f}")

    # ── Silhouette ────────────────────────────────────────────────
    print("    Computing silhouette scores (may take ~1 min)...")
    sil_zid_subj  = compute_silhouette(z_id,    y_subj)
    sil_zid_state = compute_silhouette(z_id,    y_state)
    sil_zst_subj  = compute_silhouette(z_state, y_subj)
    sil_zst_state = compute_silhouette(z_state, y_state)

    # ── Cosine statistics ──────────────────────────────────────────
    same_mask   = np.array([y_subj[i] == y_subj[j]
                            for i in range(0, min(2000, len(y_subj)), 1)
                            for j in range(i+1, min(2000, len(y_subj)), 1)
                            if i != j])
    # Simplified: pairwise cosine on subsample
    N_sub = min(2000, len(z_id))
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(z_id), N_sub, replace=False)
    z_sub = z_id[idx];  y_sub = y_subj[idx]
    cos_mat = z_sub @ z_sub.T  # (N_sub, N_sub)
    triu_idx = np.triu_indices(N_sub, k=1)
    cos_vals = cos_mat[triu_idx]
    lbl_same = (y_sub[triu_idx[0]] == y_sub[triu_idx[1]])
    intra_cos = float(np.mean(cos_vals[lbl_same]))
    inter_cos = float(np.mean(cos_vals[~lbl_same]))

    metrics = {
        "seed":                  seed,
        # Probes
        "zid_subject_probe":     {"acc": zid_subj_acc,   "std": zid_subj_std,
                                  "chance": chance_subj},
        "zid_state_probe":       {"acc": zid_state_acc,  "std": zid_state_std,
                                  "chance": chance_state},
        "zstate_subject_probe":  {"acc": zst_subj_acc,   "std": zst_subj_std,
                                  "chance": chance_subj},
        "zstate_state_probe":    {"acc": zst_state_acc,  "std": zst_state_std,
                                  "chance": chance_state},
        # MI
        "mi_zid_subject":        mi_zid_subj,
        "mi_zid_state":          mi_zid_state,
        "mi_zstate_subject":     mi_zst_subj,
        "mi_zstate_state":       mi_zst_state,
        # Orthogonality
        "orth_mean_cosine":      orth_mean,
        "orth_std_cosine":       orth_std,
        # Silhouette
        "sil_zid_subject":       sil_zid_subj,
        "sil_zid_state":         sil_zid_state,
        "sil_zstate_subject":    sil_zst_subj,
        "sil_zstate_state":      sil_zst_state,
        # Cosine similarity stats
        "zid_intra_subject_cos": intra_cos,
        "zid_inter_subject_cos": inter_cos,
        "zid_stability_gap":     intra_cos - inter_cos,
    }

    # Save raw cosine distribution for FIG_09
    np.save(str(Path(LOG_DIR) / f"seed_{seed}_cosines_zid_zstate.npy"), cosines)

    # Save embeddings for t-SNE
    np.savez(str(Path(LOG_DIR) / f"seed_{seed}_embeddings_disentangle.npz"),
             z_id=z_id, z_state=z_state, y_subj=y_subj, y_state=y_state)

    return metrics, z_id, z_state, y_subj, y_state, cosines


def plot_tsne(z_id, z_state, y_subj, y_state, seed,
              out_dir_png=None, out_dir_pdf=None):
    """FIG_08 — t-SNE of z_id coloured by subject and z_state coloured by state."""
    from sklearn.manifold import TSNE
    print("    t-SNE (may take ~2 min)...")
    if out_dir_png is None: out_dir_png = Path(PNG_DIR)
    if out_dir_pdf is None: out_dir_pdf = Path(PDF_DIR)

    # Subsample for speed
    rng = np.random.default_rng(42)
    N   = min(3000, len(z_id))
    idx = rng.choice(len(z_id), N, replace=False)
    z_id_s   = z_id[idx];    z_st_s   = z_state[idx]
    y_subj_s = y_subj[idx];  y_st_s   = y_state[idx]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, init='pca')
    emb_id = tsne.fit_transform(z_id_s)

    tsne2 = TSNE(n_components=2, perplexity=30, n_iter=1000,
                 random_state=42, init='pca')
    emb_st = tsne2.fit_transform(z_st_s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # z_id coloured by subject (plot a random 20-subject subset for clarity)
    unique_subj = np.unique(y_subj_s)
    sel_subj    = unique_subj[:20]
    cmap_s      = plt.cm.get_cmap('tab20', len(sel_subj))
    ax = axes[0]
    for k, s in enumerate(sel_subj):
        mask = y_subj_s == s
        ax.scatter(emb_id[mask, 0], emb_id[mask, 1],
                   s=5, alpha=0.6, color=cmap_s(k))
    ax.set_title(r"$z_{id}$ — coloured by subject (20 shown)")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])

    # z_state coloured by cognitive state (rest vs task)
    ax = axes[1]
    state_colors = {0: COLORS["clean"], 1: COLORS["fgsm"]}
    state_labels = {0: "Rest (R01+R02)", 1: "Task (R03–R14)"}
    for st in [0, 1]:
        mask = y_st_s == st
        if mask.sum() == 0: continue
        ax.scatter(emb_st[mask, 0], emb_st[mask, 1],
                   s=5, alpha=0.6, color=state_colors[st],
                   label=state_labels[st])
    ax.set_title(r"$z_{state}$ — coloured by cognitive state")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=8)

    fig.suptitle(f"DOMCS-EEG — Embedding Space Visualisation (Seed {seed})",
                 fontsize=10)
    fig.tight_layout()

    for d, ext in [(out_dir_png, 'png'), (out_dir_pdf, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_08_disentanglement_tsne_seed{seed}.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_08 saved")


def plot_orthogonality(cosines, seed, out_dir_png=None, out_dir_pdf=None):
    """FIG_09 — cosine(z_id, z_state) distribution."""
    if out_dir_png is None: out_dir_png = Path(PNG_DIR)
    if out_dir_pdf is None: out_dir_pdf = Path(PDF_DIR)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.hist(cosines, bins=60, color=COLORS["pgd"], edgecolor='white', alpha=0.8)
    ax.axvline(0, color='k', lw=1.0, ls='--', label='Ideal (0)')
    ax.axvline(float(np.mean(cosines)), color='red', lw=1.2, ls='-',
               label=f"Mean={np.mean(cosines):.4f}")
    ax.set_xlabel(r"$\cos(z_{id},\, z_{state})$")
    ax.set_ylabel("Count")
    ax.set_title(f"Orthogonality Distribution — Seed {seed}")
    ax.legend(fontsize=8)
    for d, ext in [(out_dir_png, 'png'), (out_dir_pdf, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_09_orthogonality_seed{seed}.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_09 saved")


def plot_probe_accuracy(metrics, out_dir_png=None, out_dir_pdf=None):
    """FIG_10 — probe accuracy comparison bar chart."""
    if out_dir_png is None: out_dir_png = Path(PNG_DIR)
    if out_dir_pdf is None: out_dir_pdf = Path(PDF_DIR)
    seed = metrics["seed"]

    labels = ['z_id→Subject', 'z_id→State', 'z_state→Subject', 'z_state→State']
    accs   = [metrics["zid_subject_probe"]["acc"],
              metrics["zid_state_probe"]["acc"],
              metrics["zstate_subject_probe"]["acc"],
              metrics["zstate_state_probe"]["acc"]]
    stds   = [metrics["zid_subject_probe"]["std"],
              metrics["zid_state_probe"]["std"],
              metrics["zstate_subject_probe"]["std"],
              metrics["zstate_state_probe"]["std"]]
    colors = [COLORS["z_id"], COLORS["z_id"], COLORS["z_state"], COLORS["z_state"]]
    chance_subj  = metrics["zid_subject_probe"]["chance"]
    chance_state = metrics["zid_state_probe"]["chance"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, [a * 100 for a in accs], yerr=[s * 100 for s in stds],
                  color=colors, alpha=0.8, capsize=4, edgecolor='white')
    ax.axhline(chance_subj  * 100, color='grey',  ls='--', lw=0.9, label=f'Chance subj ({chance_subj*100:.1f}%)')
    ax.axhline(chance_state * 100, color='orange', ls='--', lw=0.9, label=f'Chance state ({chance_state*100:.0f}%)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylabel("Probe Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title(f"Linear Probe Accuracy — Seed {seed}")
    ax.legend(fontsize=7)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=7.5, padding=2)
    for d, ext in [(out_dir_png, 'png'), (out_dir_pdf, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_10_probe_accuracy_seed{seed}.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_10 saved")


def main():
    print("DOMCS-EEG — Disentanglement Analysis")
    print("="*60)
    device = get_device()
    X, y, session = load_npz()

    all_metrics = []
    best_seed   = SEEDS[0]  # use seed 1 for figures by default

    for seed in SEEDS:
        out = run_disentanglement(seed, X, y, session, device)
        if out is None: continue
        metrics, z_id, z_state, y_subj, y_state, cosines = out
        all_metrics.append(metrics)

        # Print summary
        print(f"\n  Seed {seed} disentanglement:")
        print(f"    z_id  → subject: {metrics['zid_subject_probe']['acc']*100:.1f}%  "
              f"(chance={metrics['zid_subject_probe']['chance']*100:.1f}%)")
        print(f"    z_id  → state:   {metrics['zid_state_probe']['acc']*100:.1f}%  "
              f"(chance={metrics['zid_state_probe']['chance']*100:.0f}%)")
        print(f"    z_state→subject: {metrics['zstate_subject_probe']['acc']*100:.1f}%  "
              f"(chance={metrics['zstate_subject_probe']['chance']*100:.1f}%)")
        print(f"    z_state→state:   {metrics['zstate_state_probe']['acc']*100:.1f}%  "
              f"(chance={metrics['zstate_state_probe']['chance']*100:.0f}%)")
        print(f"    Orthogonality:   {metrics['orth_mean_cosine']:.4f}")
        print(f"    Sil(z_id,subj):  {metrics['sil_zid_subject']:.4f}")
        print(f"    Sil(z_st,state): {metrics['sil_zstate_state']:.4f}")

        # Generate figures for all seeds
        plot_tsne(z_id, z_state, y_subj, y_state, seed)
        plot_orthogonality(cosines, seed)
        plot_probe_accuracy(metrics)

    # Save aggregate metrics
    out_path = Path(LOG_DIR) / "disentanglement_metrics.json"
    with open(str(out_path), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ Disentanglement metrics saved: {out_path}")


if __name__ == "__main__":
    main()
