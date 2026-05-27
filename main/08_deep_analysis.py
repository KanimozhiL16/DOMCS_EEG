#!/usr/bin/env python3
"""
08_deep_analysis.py
====================
TWO MAJOR ANALYSES:

PART A — ArcFace + SupCon Inter/Intra Subject Similarity
  Proves identity encoder creates well-separated subject clusters.
  Produces: FIG_SUPP_A1 (similarity matrix), FIG_SUPP_A2 (distributions),
            FIG_SUPP_A3 (ArcFace logit heatmap), FIG_SUPP_A4 (SupCon pairs)

PART B — Single Segment Deep Trace (full pipeline transparency)
  Takes ONE EEG window and traces every transformation step by step.
  Produces: FIG_SUPP_B1 (preprocessed EEG), FIG_SUPP_B2 (conv layer activations),
            FIG_SUPP_B3 (embedding vectors), FIG_SUPP_B4 (scoring + EER)
"""

import sys, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as sk_auc

sys.path.insert(0, str(Path(__file__).parent))
from config import (CKPT_DIR, LOG_DIR, SUPP_DIR, PNG_DIR, PDF_DIR,
                    REST_RUNS, TASK_RUNS, MPL_STYLE)
from model       import DOMCSEEGModel, ArcFaceHead
from data_loader import load_npz, build_gallery, compute_eer, score_probe

plt.rcParams.update(MPL_STYLE)
plt.rcParams["font.size"] = 9

SUPP = Path(SUPP_DIR)
SUPP.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_best_model(device):
    import json
    eval_path = Path(LOG_DIR) / "eval_summary.json"
    with open(eval_path) as f:
        s = json.load(f)
    best_seed = s["best_seed"]
    ckpt_path = Path(CKPT_DIR) / f"seed_{best_seed}" / "model_best.pt"
    model = DOMCSEEGModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: seed_{best_seed}  EER={s['best_eer']:.4f}%")
    return model, best_seed


def savefig(fig, name):
    fig.savefig(str(SUPP / f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(str(SUPP / f"{name}.pdf"),           bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {name} saved")


# ═══════════════════════════════════════════════════════════════════
#  PART A — ARCFACE + SUPCON SIMILARITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def part_a_similarity_analysis(model, X, y, session, device,
                                n_subjects=20, n_wins_per_subj=15):
    """
    Sample n_subjects × n_wins_per_subj REST windows.
    Compute all pairwise cosine similarities of z_id.
    Show: intra-subject (diagonal blocks) vs inter-subject (off-diagonal).
    """
    print("\n  [PART A] ArcFace + SupCon similarity analysis...")

    rest_mask = np.isin(session, REST_RUNS)
    X_rest = X[rest_mask]
    y_rest = y[rest_mask]

    # Select subjects that have enough windows
    subjects  = sorted(np.unique(y_rest))
    sel_subjs = []
    for s in subjects:
        if (y_rest == s).sum() >= n_wins_per_subj:
            sel_subjs.append(s)
        if len(sel_subjs) == n_subjects:
            break

    rng = np.random.default_rng(42)
    X_sel, y_sel = [], []
    for s in sel_subjs:
        idx = np.where(y_rest == s)[0]
        chosen = rng.choice(idx, n_wins_per_subj, replace=False)
        X_sel.append(X_rest[chosen])
        y_sel.extend([s] * n_wins_per_subj)

    X_sel = np.concatenate(X_sel, axis=0)  # (n_subjects*n_wins, 64, 256)
    y_sel = np.array(y_sel)

    # Extract z_id for all selected windows
    with torch.no_grad():
        xb   = torch.from_numpy(X_sel).to(device)
        z_id = model.get_identity_embedding(xb).cpu().numpy()  # (N, 128)

    N = len(z_id)
    # Cosine similarity matrix (already L2-normalised, so dot product = cosine)
    sim_matrix = z_id @ z_id.T   # (N, N)

    # ── FIG_SUPP_A1: Pairwise Similarity Matrix ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full similarity matrix
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                   aspect='auto')
    # Draw subject boundaries
    for i in range(1, n_subjects):
        pos = i * n_wins_per_subj - 0.5
        ax.axhline(pos, color='k', linewidth=0.4, alpha=0.7)
        ax.axvline(pos, color='k', linewidth=0.4, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'Pairwise Cosine Similarity Matrix\n'
                 f'({n_subjects} subjects × {n_wins_per_subj} REST windows each)',
                 fontsize=10)
    ax.set_xlabel('Window index (grouped by subject)')
    ax.set_ylabel('Window index (grouped by subject)')

    # Intra vs Inter distribution
    ax2 = axes[1]
    intra, inter = [], []
    for i in range(N):
        for j in range(i+1, N):
            v = sim_matrix[i, j]
            if y_sel[i] == y_sel[j]:
                intra.append(v)
            else:
                inter.append(v)

    intra = np.array(intra)
    inter = np.array(inter)

    bins = np.linspace(-0.3, 1.05, 60)
    ax2.hist(inter, bins=bins, alpha=0.6, color='#d62728', density=True,
             label=f'Inter-subject (n={len(inter):,})\nμ={inter.mean():.3f}±{inter.std():.3f}')
    ax2.hist(intra, bins=bins, alpha=0.6, color='#2ca02c', density=True,
             label=f'Intra-subject (n={len(intra):,})\nμ={intra.mean():.3f}±{intra.std():.3f}')

    # Vertical lines for means
    ax2.axvline(inter.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    ax2.axvline(intra.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)

    # D-prime annotation
    dprime = (intra.mean() - inter.mean()) / \
             np.sqrt(0.5*(intra.std()**2 + inter.std()**2))
    ax2.text(0.05, 0.92, f"d' = {dprime:.3f}", transform=ax2.transAxes,
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_xlabel('Cosine similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('ArcFace+SupCon: Intra vs Inter-Subject\nCosine Similarity Distribution')
    ax2.legend(fontsize=8)

    fig.suptitle('DOMCS-EEG: Identity Embedding Similarity Analysis\n'
                 '(Enrollment REST windows, best-seed model)', fontsize=11)
    plt.tight_layout()
    savefig(fig, "FIG_SUPP_A1_similarity_matrix_and_distribution")

    # ── FIG_SUPP_A2: Per-Subject Intra-Similarity Boxplot ────────
    fig, ax = plt.subplots(figsize=(14, 4))
    per_subj_intra = []
    for s in sel_subjs:
        idx = np.where(y_sel == s)[0]
        z_s = z_id[idx]
        sm  = z_s @ z_s.T
        # Upper triangle excluding diagonal
        vals = sm[np.triu_indices(len(z_s), k=1)]
        per_subj_intra.append(vals)

    bp = ax.boxplot(per_subj_intra, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch in bp['boxes']:
        patch.set_facecolor('#2ca02c')
        patch.set_alpha(0.6)

    ax.axhline(inter.mean(), color='#d62728', linestyle='--',
               linewidth=1.5, label=f'Inter-subject mean={inter.mean():.3f}')
    ax.set_xlabel('Subject index (S1–S20 REST windows)')
    ax.set_ylabel('Intra-subject cosine similarity')
    ax.set_title('Per-Subject Intra-Subject Cosine Similarity\n'
                 '(ArcFace+SupCon forces compact clusters above inter-subject mean)')
    ax.legend()
    plt.tight_layout()
    savefig(fig, "FIG_SUPP_A2_per_subject_intra_similarity")

    # ── Print statistics ─────────────────────────────────────────
    from scipy import stats as sp_stats
    t_stat, p_val = sp_stats.ttest_ind(intra, inter, alternative='greater')
    print(f"\n  INTRA-SUBJECT SIMILARITY: μ={intra.mean():.4f} ± {intra.std():.4f}")
    print(f"  INTER-SUBJECT SIMILARITY: μ={inter.mean():.4f} ± {inter.std():.4f}")
    print(f"  D-PRIME:                  d'={dprime:.4f}")
    print(f"  T-TEST (intra > inter):   t={t_stat:.2f}  p={p_val:.2e}")
    print(f"  Gap (intra-inter mean):   {intra.mean()-inter.mean():.4f}")

    return intra, inter, z_id, y_sel, sel_subjs


# ═══════════════════════════════════════════════════════════════════
#  PART B — SINGLE SEGMENT DEEP TRACE
# ═══════════════════════════════════════════════════════════════════

def part_b_single_segment_trace(model, X, y, session, gallery, device,
                                 best_seed,
                                 subject_id=0, run='R01', window_idx=0):
    """
    Full step-by-step trace of ONE EEG window through the entire pipeline.
    """
    print(f"\n  [PART B] Deep trace: subject={subject_id} run={run} window={window_idx}")

    # ── Select the segment ────────────────────────────────────────
    mask = (y == subject_id) & (session == run)
    indices = np.where(mask)[0]
    assert len(indices) > window_idx, f"Not enough windows for subject {subject_id} in {run}"
    seg_idx = indices[window_idx]
    x_raw   = X[seg_idx]      # (64, 256) float32
    true_subj = int(y[seg_idx])

    print(f"  Segment index in dataset: {seg_idx}")
    print(f"  True subject: S{true_subj+1:03d}  Run: {run}  Window: {window_idx}")
    print(f"  Input shape: {x_raw.shape}  Range: [{x_raw.min():.4f}, {x_raw.max():.4f}]")

    # ── FIG_SUPP_B1: Preprocessed EEG Input ─────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig)

    # B1-a: Full EEG heatmap (64 channels × 256 samples)
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(x_raw, cmap='RdBu_r', aspect='auto',
                     vmin=-3, vmax=3,
                     extent=[0, 256/128, 0, 64])
    plt.colorbar(im1, ax=ax1, label='z-scored amplitude')
    ax1.set_title(f'STEP 1: Preprocessed EEG Input (z-scored, 2s window @ 128Hz)\n'
                  f'Subject S{true_subj+1:03d} | Run {run} | Window {window_idx}\n'
                  f'Shape: (64 channels × 256 samples = 2s @ 128Hz)',
                  fontsize=9)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('EEG Channel index')

    # B1-b: Sample 4 individual channel traces
    ax2 = fig.add_subplot(gs[0, 2])
    t = np.arange(256) / 128
    chs = [0, 16, 32, 48]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (ch, c) in enumerate(zip(chs, colors)):
        ax2.plot(t, x_raw[ch] + i*4, color=c, linewidth=0.8,
                 label=f'Ch{ch+1}')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Sample 4 channels\n(offset for clarity)')
    ax2.legend(fontsize=7, loc='upper right')

    # B1-c: Channel power spectrum (mean across channels)
    ax3 = fig.add_subplot(gs[1, 0])
    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(256, 1/128)
    psd = np.abs(rfft(x_raw, axis=1))**2
    mean_psd = psd.mean(axis=0)
    ax3.semilogy(freqs[:40], mean_psd[:40], color='#1f77b4')
    ax3.axvspan(8, 13,  alpha=0.15, color='green',  label='Alpha (8-13Hz)')
    ax3.axvspan(13, 30, alpha=0.15, color='orange', label='Beta (13-30Hz)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.set_title('Mean Power Spectrum')
    ax3.legend(fontsize=7)

    # B1-d: Channel variance across time (topographic proxy)
    ax4 = fig.add_subplot(gs[1, 1])
    ch_var = x_raw.var(axis=1)
    ax4.bar(np.arange(64), ch_var, color='#1f77b4', alpha=0.7, width=0.8)
    ax4.set_xlabel('Channel index')
    ax4.set_ylabel('Temporal variance')
    ax4.set_title('Per-Channel Temporal Variance')

    # B1-e: Summary stats
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats_text = (
        f"SEGMENT STATISTICS\n"
        f"{'─'*28}\n"
        f"Subject:    S{true_subj+1:03d} (ID={true_subj})\n"
        f"Run:        {run}\n"
        f"Shape:      64 ch × 256 samples\n"
        f"Duration:   2.0 seconds\n"
        f"Fs:         128 Hz\n"
        f"Window step: 1s (50% overlap)\n\n"
        f"AMPLITUDE STATS\n"
        f"{'─'*28}\n"
        f"Min:   {x_raw.min():.4f}\n"
        f"Max:   {x_raw.max():.4f}\n"
        f"Mean:  {x_raw.mean():.4f}\n"
        f"Std:   {x_raw.std():.4f}\n"
        f"(z-scored → mean≈0, std≈1)"
    )
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
             fontsize=8, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    fig.suptitle('PART B — Single Segment Deep Trace\nSTEP 1: PREPROCESSED EEG INPUT (z-scored, 2s window @ 128Hz)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    savefig(fig, "FIG_SUPP_B1_preprocessed_eeg_input")

    # ── Register hooks for intermediate activations ───────────────
    activations = {}
    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    h1 = model.encoder.conv1.register_forward_hook(make_hook('conv1'))
    h2 = model.encoder.conv2.register_forward_hook(make_hook('conv2'))
    h3 = model.encoder.conv3.register_forward_hook(make_hook('conv3'))
    h4 = model.encoder.pool.register_forward_hook(make_hook('pool'))

    x_tensor = torch.from_numpy(x_raw).unsqueeze(0).to(device)  # (1,64,256)
    with torch.no_grad():
        z_id_t, z_state_t, f_t = model(x_tensor)

    h1.remove(); h2.remove(); h3.remove(); h4.remove()

    conv1_out = activations['conv1'].squeeze(0).numpy()   # (64,  256)
    conv2_out = activations['conv2'].squeeze(0).numpy()   # (128, 256)
    conv3_out = activations['conv3'].squeeze(0).numpy()   # (256, 256)
    pool_out  = activations['pool'].squeeze(0).squeeze(-1).numpy()  # (256,)
    f_vec     = f_t.squeeze(0).cpu().numpy()              # (256,)
    z_id_vec  = z_id_t.squeeze(0).cpu().numpy()           # (128,)
    z_st_vec  = z_state_t.squeeze(0).cpu().numpy()        # (128,)

    print(f"\n  LAYER OUTPUTS:")
    print(f"    Input:     {x_raw.shape}   range [{x_raw.min():.3f}, {x_raw.max():.3f}]")
    print(f"    Conv1:     {conv1_out.shape}  range [{conv1_out.min():.3f}, {conv1_out.max():.3f}]")
    print(f"    Conv2:     {conv2_out.shape} range [{conv2_out.min():.3f}, {conv2_out.max():.3f}]")
    print(f"    Conv3:     {conv3_out.shape} range [{conv3_out.min():.3f}, {conv3_out.max():.3f}]")
    print(f"    Pool→f:    {f_vec.shape}     range [{f_vec.min():.3f}, {f_vec.max():.3f}]")
    print(f"    z_id:      {z_id_vec.shape}    L2-norm={np.linalg.norm(z_id_vec):.6f}")
    print(f"    z_state:   {z_st_vec.shape}    L2-norm={np.linalg.norm(z_st_vec):.6f}")

    # ── FIG_SUPP_B2: Layer-by-Layer Activations ──────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Input — 2D heatmap (64 channels × 256 samples) ✓
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(x_raw, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax.set_title(f'INPUT\n(64×256)\nRange:[{x_raw.min():.2f},{x_raw.max():.2f}]',
                 fontsize=8)
    ax.set_ylabel('Channel'); ax.set_xlabel('Time sample')

    # Conv1 — show 4 individual filter activations as TIME SERIES (1D → plot not imshow)
    filter_indices = [0, 16, 32, 48]
    line_colors    = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
    subplot_pos    = [(0,1),(0,2),(0,3),(1,0)]
    for fi, (filt_idx, col, pos) in enumerate(zip(filter_indices, line_colors, subplot_pos)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        t_ax = np.arange(256) / 128.0
        # conv1_out[filt_idx] is 1D (256,) — one filter's temporal response
        ax.plot(t_ax, conv1_out[filt_idx], color=col, linewidth=0.8)
        ax.fill_between(t_ax, conv1_out[filt_idx], alpha=0.2, color=col)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax.set_title(f'Conv1 filter {filt_idx} (of 64)\n'
                     f'k=7, BN, ELU | out shape: (64×256)\n'
                     f'range:[{conv1_out[filt_idx].min():.2f},{conv1_out[filt_idx].max():.2f}]',
                     fontsize=7)
        ax.set_xlabel('Time (s)')

    # Conv2 — 2D heatmap of first 32 filters (shape 32×256) ✓
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(conv2_out[:32], cmap='viridis', aspect='auto')
    ax.set_title(f'Conv2 (first 32 of 128 filters)\n(k=5,BN,ELU)\nfull out:(128×256)',
                 fontsize=8)
    ax.set_ylabel('Filter index'); ax.set_xlabel('Time sample')

    # Conv3 — 2D heatmap of first 32 filters (shape 32×256) ✓
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(conv3_out[:32], cmap='viridis', aspect='auto')
    ax.set_title(f'Conv3 (first 32 of 256 filters)\n(k=3,BN,ELU)\nfull out:(256×256)',
                 fontsize=8)
    ax.set_ylabel('Filter index'); ax.set_xlabel('Time sample')

    # Activation statistics per layer
    ax = fig.add_subplot(gs[1, 3])
    layers_names = ['Input', 'Conv1', 'Conv2', 'Conv3']
    layers_data  = [x_raw, conv1_out, conv2_out, conv3_out]
    means  = [d.mean() for d in layers_data]
    stds   = [d.std()  for d in layers_data]
    x_pos  = np.arange(len(layers_names))
    ax.bar(x_pos, means, yerr=stds, capsize=4, color=['#1f77b4','#ff7f0e','#2ca02c','#d62728'],
           alpha=0.8)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(layers_names, fontsize=8)
    ax.set_ylabel('Mean activation')
    ax.set_title('Activation Statistics\nper Layer (mean±std)')

    # f vector (256,) — encoder output
    ax = fig.add_subplot(gs[2, :2])
    ax.bar(np.arange(256), f_vec, width=1.0, color='#1f77b4', alpha=0.7)
    ax.set_title(f'STEP 5: Encoder output f ∈ ℝ^256\n'
                 f'(After AdaptiveAvgPool — shared representation)\n'
                 f'Mean={f_vec.mean():.4f}  Std={f_vec.std():.4f}  '
                 f'Range=[{f_vec.min():.3f},{f_vec.max():.3f}]',
                 fontsize=8)
    ax.set_xlabel('Dimension index')
    ax.set_ylabel('Activation value')

    # z_id (128,) — identity embedding
    ax = fig.add_subplot(gs[2, 2])
    colors_id = ['#2ca02c' if v > 0 else '#d62728' for v in z_id_vec]
    ax.bar(np.arange(128), z_id_vec, width=1.0, color=colors_id, alpha=0.8)
    ax.set_title(f'z_id ∈ ℝ^128 (Identity)\n'
                 f'L2-norm={np.linalg.norm(z_id_vec):.6f} (should=1.0)\n'
                 f'[Linear→LayerNorm→L2-normalize]',
                 fontsize=8)
    ax.set_xlabel('Dimension'); ax.set_ylabel('Value')

    # z_state (128,) — state embedding
    ax = fig.add_subplot(gs[2, 3])
    colors_st = ['#9467bd' if v > 0 else '#e377c2' for v in z_st_vec]
    ax.bar(np.arange(128), z_st_vec, width=1.0, color=colors_st, alpha=0.8)
    ax.set_title(f'z_state ∈ ℝ^128 (Cognitive State)\n'
                 f'L2-norm={np.linalg.norm(z_st_vec):.6f} (should=1.0)\n'
                 f'[f.detach()→Linear→LayerNorm→L2-normalize]',
                 fontsize=8)
    ax.set_xlabel('Dimension'); ax.set_ylabel('Value')

    # Orthogonality check
    ax = fig.add_subplot(gs[3, 0])
    cos_sim = float(np.dot(z_id_vec, z_st_vec))
    ax.bar(['cos(z_id, z_state)'], [cos_sim],
           color='#17becf' if abs(cos_sim) < 0.05 else '#d62728')
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylim(-0.3, 0.3)
    ax.set_title(f'Orthogonality Check\ncos(z_id, z_state)={cos_sim:.6f}\n'
                 f'(≈0 = disentangled ✓)')

    # Cosine similarity between z_id and each dimension of f
    ax = fig.add_subplot(gs[3, 1])
    f_norm = f_vec / (np.linalg.norm(f_vec) + 1e-8)
    # Projection of z_id and z_state onto f
    proj_id = np.dot(z_id_vec, f_norm[:128] / (np.linalg.norm(f_norm[:128])+1e-8))
    proj_st = np.dot(z_st_vec, f_norm[128:] / (np.linalg.norm(f_norm[128:])+1e-8))
    ax.bar(['z_id·f[:128]', 'z_st·f[128:]'],
           [proj_id, proj_st],
           color=['#2ca02c','#9467bd'], alpha=0.8)
    ax.set_title('Embedding-Encoder Alignment\n(cosine with encoder sub-space)')
    ax.set_ylabel('Cosine similarity')

    # Pipeline summary text
    ax = fig.add_subplot(gs[3, 2:])
    ax.axis('off')
    pipeline_text = (
        "COMPLETE PIPELINE STEPS\n"
        "─────────────────────────────────────────\n"
        "1. INPUT:  (1, 64, 256) z-scored EEG\n"
        "2. CONV1:  Conv1d(64→64,k=7,p=3)→BN→ELU → (64,256)\n"
        "3. CONV2:  Conv1d(64→128,k=5,p=2)→BN→ELU → (128,256)\n"
        "4. CONV3:  Conv1d(128→256,k=3,p=1)→BN→ELU → (256,256)\n"
        "5. POOL:   AdaptiveAvgPool1d(1) → f∈ℝ^256\n"
        "── IDENTITY BRANCH ──\n"
        "6. ID:     Linear(256→128)→LayerNorm→L2-norm → z_id∈ℝ^128\n"
        "── STATE BRANCH (detached) ──\n"
        "7. STATE:  f.detach()→Linear(256→128)→LN→L2-norm → z_state\n"
        "── VERIFICATION ──\n"
        "8. GALLERY: KMeans(K=3) on REST z_id → 3 prototypes/subject\n"
        "9. SCORE:  max cosine sim(z_id, gallery_prototypes)\n"
        "10. EER:   roc_curve → equal FPR/FNR threshold"
    )
    ax.text(0.02, 0.98, pipeline_text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fffff0', alpha=0.9))

    fig.suptitle('PART B — STEP 2–7: CNN Layer-by-Layer Activations\n'
                 f'Subject S{true_subj+1:03d} | Run {run} | Window {window_idx}',
                 fontsize=12, fontweight='bold')
    savefig(fig, "FIG_SUPP_B2_layer_activations")

    # ── FIG_SUPP_B3: Gallery Scoring Step ────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Score against all subjects
    z_id_norm = z_id_vec / (np.linalg.norm(z_id_vec) + 1e-8)
    all_scores = score_probe(z_id_norm, gallery)
    all_subj_ids  = sorted(all_scores.keys())
    all_score_vals = np.array([all_scores[s] for s in all_subj_ids])

    # Sort by score descending
    sort_idx = np.argsort(all_score_vals)[::-1]
    sorted_ids   = np.array(all_subj_ids)[sort_idx]
    sorted_scores= all_score_vals[sort_idx]

    genuine_score = all_scores[true_subj]
    rank = int(np.where(sorted_ids == true_subj)[0][0]) + 1

    print(f"\n  SCORING RESULTS:")
    print(f"    Genuine score (S{true_subj+1:03d}): {genuine_score:.4f}")
    print(f"    Genuine rank: {rank} of {len(all_subj_ids)}")
    print(f"    Top-5 scores: {sorted_scores[:5].round(4)}")
    print(f"    Top-5 subj IDs: {sorted_ids[:5]+1}")

    # B3-a: Score against all 109 subjects (bar chart)
    ax = fig.add_subplot(gs[0, :2])
    bar_colors = ['#2ca02c' if s == true_subj else '#1f77b4' for s in all_subj_ids]
    ax.bar(np.arange(len(all_subj_ids)), all_score_vals, width=1.0,
           color=bar_colors, alpha=0.8)
    ax.axhline(genuine_score, color='#2ca02c', linestyle='--', linewidth=1.2,
               label=f'Genuine score (S{true_subj+1:03d})={genuine_score:.4f}')

    # Mark genuine subject
    genuine_pos = all_subj_ids.index(true_subj)
    ax.bar(genuine_pos, all_score_vals[genuine_pos],
           color='#2ca02c', alpha=1.0, label=f'True subject S{true_subj+1:03d}')
    ax.set_xlabel('Subject ID (0–108)')
    ax.set_ylabel('Max cosine similarity to gallery')
    ax.set_title(f'STEP 9: Gallery Scoring — Probe vs All 109 Gallery Subjects\n'
                 f'True subject S{true_subj+1:03d} | Rank={rank} | '
                 f'Genuine={genuine_score:.4f}')
    ax.legend(fontsize=8)

    # B3-b: Top-20 scores detail
    ax = fig.add_subplot(gs[0, 2])
    top20_ids    = sorted_ids[:20]
    top20_scores = sorted_scores[:20]
    top20_colors = ['#2ca02c' if s == true_subj else '#d62728' for s in top20_ids]
    ax.barh(np.arange(20), top20_scores[::-1], color=top20_colors[::-1], alpha=0.8)
    yticks = [f"S{s+1:03d}{'✓' if s==true_subj else ''}" for s in top20_ids[::-1]]
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(yticks, fontsize=7)
    ax.set_xlabel('Cosine similarity')
    ax.set_title('Top-20 Scoring Subjects\n(green=true, red=impostor)')

    # B3-c: Gallery prototypes for true subject
    ax = fig.add_subplot(gs[1, 0])
    protos = gallery[true_subj]   # (K, 128)
    K = len(protos)
    for k in range(K):
        ax.plot(protos[k], alpha=0.7, label=f'Prototype {k+1}', linewidth=0.8)
    ax.plot(z_id_vec, 'k--', linewidth=1.2, label='Query z_id', alpha=0.9)
    ax.set_title(f'Gallery Prototypes for S{true_subj+1:03d}\nvs Query z_id\n(K={K} KMeans prototypes from REST)')
    ax.set_xlabel('Embedding dimension')
    ax.legend(fontsize=7)

    # B3-d: Cosine similarities to each prototype
    ax = fig.add_subplot(gs[1, 1])
    proto_sims = [float(np.dot(z_id_norm, p)) for p in protos]
    ax.bar(np.arange(K), proto_sims, color='#2ca02c', alpha=0.8)
    ax.set_xticks(np.arange(K))
    ax.set_xticklabels([f'Proto {k+1}' for k in range(K)])
    ax.set_ylabel('Cosine similarity')
    ax.set_title(f'Similarity to Each Prototype\nMax={max(proto_sims):.4f} (used as genuine score)')
    for k, v in enumerate(proto_sims):
        ax.text(k, v+0.005, f'{v:.4f}', ha='center', fontsize=8)

    # B3-e: Accept/Reject decision annotation
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    # Load EER threshold from scores
    score_path = Path(LOG_DIR) / f"seed_{best_seed}" / f"scores_seed{best_seed}.npz"
    eer_thresh = None
    if score_path.exists():
        sc  = np.load(str(score_path))
        gen = sc["genuine"]; imp = sc["impostor"]
        y_true  = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
        y_score = np.concatenate([gen, imp])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer_thresh = float(thr[idx])

    decision = "ACCEPT ✓" if genuine_score >= (eer_thresh or 0.5) else "REJECT ✗"
    dec_color = '#2ca02c' if 'ACCEPT' in decision else '#d62728'
    decision_text = (
        f"VERIFICATION DECISION\n"
        f"{'─'*30}\n"
        f"Query: Subject S{true_subj+1:03d}\n"
        f"Run: {run} Window: {window_idx}\n\n"
        f"Genuine score:  {genuine_score:.4f}\n"
        f"EER threshold: {f'{eer_thresh:.4f}' if eer_thresh is not None else 'N/A'}\n"
        f"Rank among 109: #{rank}\n\n"
        f"DECISION: {decision}\n\n"
        f"Interpretation:\n"
        f"Score > thresh → ACCEPT\n"
        f"Score < thresh → REJECT\n\n"
        f"This probe is {'correctly' if 'ACCEPT' in decision else 'incorrectly'}\n"
        f"verified."
    )
    ax.text(0.1, 0.95, decision_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace', color='black',
            bbox=dict(boxstyle='round', facecolor=dec_color, alpha=0.15,
                      edgecolor=dec_color, linewidth=2))

    fig.suptitle('PART B — STEP 8–10: Gallery Build → Score → Decision\n'
                 f'Subject S{true_subj+1:03d} | Run {run} | Window {window_idx}',
                 fontsize=12, fontweight='bold')
    savefig(fig, "FIG_SUPP_B3_gallery_scoring_and_decision")

    # ── FIG_SUPP_B4: ROC and EER for this probe ──────────────────
    if score_path.exists():
        sc  = np.load(str(score_path))
        gen = sc["genuine"]; imp = sc["impostor"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # ROC curve
        ax = axes[0]
        ax.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'ROC (AUC={sk_auc(fpr,tpr):.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random (AUC=0.5)')
        # Mark EER point
        ax.scatter(fpr[idx], tpr[idx], color='red', s=100, zorder=5,
                   label=f'EER point ({(fpr[idx]+1-tpr[idx])/2*100:.2f}%)\nthresh={eer_thresh:.4f}')
        ax.set_xlabel('False Positive Rate (FAR)')
        ax.set_ylabel('True Positive Rate (1-FRR)')
        ax.set_title('ROC Curve\n(Seed 3, all 160,123 probes)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Score distributions with EER threshold
        ax = axes[1]
        bins = np.linspace(min(gen.min(), imp.min()),
                           max(gen.max(), imp.max()), 80)
        ax.hist(imp, bins=bins, alpha=0.6, density=True,
                color='#d62728', label=f'Impostor (n={len(imp):,})')
        ax.hist(gen, bins=bins, alpha=0.6, density=True,
                color='#2ca02c', label=f'Genuine (n={len(gen):,})')
        ax.axvline(eer_thresh, color='k', linestyle='--', linewidth=1.5,
                   label=f'EER threshold={eer_thresh:.4f}')
        # Mark this probe's genuine score
        ax.axvline(genuine_score, color='blue', linestyle='-', linewidth=2,
                   label=f'This probe score={genuine_score:.4f}')
        ax.set_xlabel('Cosine similarity score')
        ax.set_ylabel('Density')
        ax.set_title('Genuine vs Impostor Distributions\n(this probe marked in blue)')
        ax.legend(fontsize=7)

        # DET curve
        ax = axes[2]
        ax.plot(fpr*100, (1-tpr)*100, 'b-', linewidth=2)
        ax.scatter(fpr[idx]*100, (1-tpr[idx])*100, color='red', s=100, zorder=5,
                   label=f'EER={(fpr[idx]+1-tpr[idx])/2*100:.4f}%')
        ax.set_xlabel('False Acceptance Rate (%)')
        ax.set_ylabel('False Rejection Rate (%)')
        ax.set_title('DET Curve\n(symmetric at EER point)')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 20); ax.set_ylim(0, 20)
        ax.plot([0, 20], [0, 20], 'k--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)

        fig.suptitle('PART B — STEP 10: EER Computation\n'
                     f'Seed 3 (best) | 160,123 verification probes | '
                     f'This probe: S{true_subj+1:03d} score={genuine_score:.4f}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        savefig(fig, "FIG_SUPP_B4_roc_det_eer_with_probe")

    return z_id_vec, z_st_vec, genuine_score


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("DOMCS-EEG — Deep Analysis (ArcFace/SupCon + Single Segment Trace)")
    print("=" * 65)

    device = get_device()
    print(f"  Device: {device}")

    print("\nLoading dataset...")
    X, y, session = load_npz()

    print("\nLoading best model...")
    model, best_seed = load_best_model(device)

    print("\nBuilding gallery (R01+R02)...")
    gallery = build_gallery(model, X, y, session, device)
    print(f"  Gallery: {len(gallery)} subjects × {list(gallery.values())[0].shape[0]} prototypes")

    # ── PART A: ArcFace + SupCon similarity analysis ──────────────
    print("\n" + "="*65)
    print("PART A: ArcFace + SupCon Similarity Analysis")
    print("="*65)
    intra, inter, z_ids, y_sel, sel_subjs = part_a_similarity_analysis(
        model, X, y, session, device,
        n_subjects=20, n_wins_per_subj=15
    )

    # ── PART B: Single segment deep trace ────────────────────────
    print("\n" + "="*65)
    print("PART B: Single Segment Deep Trace")
    print("="*65)
    # Trace subject 0, R01, window 5 (skip first few which may be edge effects)
    z_id_vec, z_st_vec, genuine_score = part_b_single_segment_trace(
        model, X, y, session, gallery, device,
        best_seed=best_seed,
        subject_id=0, run='R01', window_idx=5
    )

    print(f"\n{'='*65}")
    print("  DEEP ANALYSIS COMPLETE")
    print(f"  All supplementary figures saved to: {SUPP}")
    print(f"\n  Files generated:")
    for f in sorted(Path(SUPP).glob("FIG_SUPP_*.png")):
        print(f"    {f.name}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
