"""
BED Dataset Visualization — For Paper / Reviewer Understanding
==============================================================
Generates 6 figures covering:
  Fig 1 — Dataset overview (window/subject/session distribution)
  Fig 2 — Stimulus-type breakdown per session (REST vs TASK)
  Fig 3 — BED-B2T-CS protocol diagram
  Fig 4 — Sample raw EEG windows: REST (r01) vs TASK (r03), 1 subject
  Fig 5 — Power Spectral Density comparison: REST vs TASK
  Fig 6 — Per-subject window balance across sessions

Run on Brev:
  python bed_visualize.py
Output saved to /home/nvidia/24PHD1237/BED_VIZ/
"""

import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.signal import welch
from collections import Counter
from pathlib import Path

# ── Argument parsing ─────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="BED Dataset Visualization — generates 6 figures for paper/reviewers"
    )
    p.add_argument(
        "--data", type=Path,
        default=Path("./data/BED_win2s_step1s_fs128.npz"),
        help="Path to BED NPZ file"
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("./BED_VIZ"),
        help="Output directory for figures"
    )
    return p.parse_args()

_args   = _parse_args()
BED_NPZ = _args.data
OUT_DIR = _args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 128          # sampling frequency
WIN_SEC = 2       # window length in seconds
WIN_SAMPLES = 256 # samples per window (FS * WIN_SEC)

# ── Palette ──────────────────────────────────────────────────────
C_REST  = "#4878CF"   # blue  — r01/r02
C_TASK  = "#D65F5F"   # red   — r03
C_R01   = "#6BAED6"
C_R02   = "#2171B5"
C_R03   = "#D65F5F"
GREY    = "#AAAAAA"

print("Loading BED NPZ …")
d = np.load(BED_NPZ, allow_pickle=True)
X        = d['X']           # (18419, 14, 256)
y        = d['y']           # subject labels  (int)
sessions = d['session']     # 'r01' / 'r02' / 'r03'
stimulus = d['stimulus']    # per-window stimulus string
channels = list(d['channels'])
subjects = sorted(set(y.tolist()))
print(f"  X shape: {X.shape}, {len(subjects)} subjects, sessions: {sorted(set(sessions.tolist()))}")

# ── Helper: simplify stimulus label ──────────────────────────────
def simplify_stim(s):
    s = str(s)
    if 'rest' in s:            return 'REST'
    if 'open_closed' in s or 'eyes' in s: return 'EYES OPEN/CLOSED'
    if 'ssvep' in s and 'c' not in s.split('ssvep')[1][:2]: return 'SSVEP'
    if 'ssvepc' in s:          return 'SSVEP-CONTROL'
    if 'stimuli image' in s or 'image' in s: return 'IMAGE VIEWING'
    return 'OTHER'

simp_stim = np.array([simplify_stim(s) for s in stimulus])

# ─────────────────────────────────────────────────────────────────
# FIG 1: Dataset overview — bar chart (sessions) + donut (subjects)
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 1 — Dataset overview …")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("BED Dataset — Overview", fontsize=14, fontweight='bold', y=1.02)

# 1a: Window count per session
sess_counts = {s: int((sessions == s).sum()) for s in ['r01','r02','r03']}
ax = axes[0]
bars = ax.bar(['r01\n(REST)', 'r02\n(REST)', 'r03\n(TASK)'],
              [sess_counts['r01'], sess_counts['r02'], sess_counts['r03']],
              color=[C_R01, C_R02, C_R03], width=0.5, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, [sess_counts['r01'], sess_counts['r02'], sess_counts['r03']]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
            f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Windows', fontsize=11)
ax.set_title('(a) Windows per Session', fontsize=12)
ax.set_ylim(0, 7200)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)

# 1b: Subject distribution (per session average)
ax = axes[1]
windows_per_sub = []
for sid in subjects:
    mask = y == sid
    windows_per_sub.append(mask.sum())
ax.hist(windows_per_sub, bins=10, color=C_REST, edgecolor='white', linewidth=1.2)
ax.axvline(np.mean(windows_per_sub), color='navy', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(windows_per_sub):.0f}')
ax.set_xlabel('Total Windows per Subject', fontsize=11)
ax.set_ylabel('Number of Subjects', fontsize=11)
ax.set_title('(b) Windows per Subject', fontsize=12)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 1c: EEG channel info
ax = axes[2]
ax.axis('off')
info_text = (
    "BED Dataset Summary\n"
    "─────────────────────────\n"
    f"  Total windows  : {len(X):,}\n"
    f"  Subjects        : {len(subjects)}\n"
    f"  EEG channels    : 14 (Emotiv EPOC+)\n"
    f"  Sampling rate   : 128 Hz\n"
    f"  Window length   : 2 s (256 samples)\n"
    f"  Window step     : 1 s\n"
    "─────────────────────────\n"
    "  Session split:\n"
    f"  r01 (REST)  : {sess_counts['r01']:,} windows\n"
    f"  r02 (REST)  : {sess_counts['r02']:,} windows\n"
    f"  r03 (TASK)  : {sess_counts['r03']:,} windows\n"
    "─────────────────────────\n"
    "  Protocol: BED-B2T-CS\n"
    "  Enroll: r01 + r02\n"
    "  Probe : r03\n"
)
ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f4f8', alpha=0.8))
ax.set_title('(c) Dataset Statistics', fontsize=12)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG1_dataset_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG1_dataset_overview.png")

# ─────────────────────────────────────────────────────────────────
# FIG 2: Stimulus-type breakdown per session
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 2 — Stimulus breakdown …")
stim_categories = ['REST', 'EYES OPEN/CLOSED', 'IMAGE VIEWING', 'SSVEP', 'SSVEP-CONTROL']
colors_stim = ['#6BAED6', '#9ECAE1', '#D65F5F', '#FC8D59', '#FDAE6B']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BED Dataset — Stimulus Distribution per Session\n"
             "(r01/r02 = REST Baseline | r03 = TASK Probes)", fontsize=13, fontweight='bold')

for ax, sess, color_base in zip(axes, ['r01','r02','r03'], [C_R01, C_R02, C_R03]):
    mask = sessions == sess
    stims_in_sess = simp_stim[mask]
    counts = Counter(stims_in_sess)
    labels = [c for c in stim_categories if c in counts]
    sizes  = [counts[c] for c in labels]
    c_map  = dict(zip(stim_categories, colors_stim))
    wedge_colors = [c_map[l] for l in labels]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%',
        colors=wedge_colors, startangle=90,
        pctdistance=0.75, wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)

    state_label = 'REST Session' if sess in ['r01','r02'] else 'TASK Session'
    ax.set_title(f'{sess} — {state_label}\n({(sessions==sess).sum():,} windows)',
                 fontsize=12, fontweight='bold',
                 color=C_REST if sess != 'r03' else C_TASK)
    ax.legend(labels, loc='lower center', bbox_to_anchor=(0.5, -0.20),
              fontsize=8, ncol=2, framealpha=0.7)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG2_stimulus_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG2_stimulus_breakdown.png")

# ─────────────────────────────────────────────────────────────────
# FIG 3: BED-B2T-CS Protocol diagram
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 3 — Protocol diagram …")
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

title = ax.text(7, 6.6, "BED-B2T-CS Protocol: Cross-Dataset Biometric Evaluation",
                ha='center', va='center', fontsize=14, fontweight='bold')

# ── Enrollment box (left) ─────────────────────────────
enroll_box = FancyBboxPatch((0.3, 3.2), 5.2, 3.1,
    boxstyle="round,pad=0.1", facecolor='#EBF5FB', edgecolor=C_REST, linewidth=2.5)
ax.add_patch(enroll_box)
ax.text(2.9, 6.0, "ENROLLMENT SET", ha='center', fontsize=12, fontweight='bold', color=C_REST)

r01_box = FancyBboxPatch((0.6, 4.4), 2.1, 1.5,
    boxstyle="round,pad=0.1", facecolor=C_R01, edgecolor='white', linewidth=1.5, alpha=0.85)
ax.add_patch(r01_box)
ax.text(1.65, 5.35, "r01", ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(1.65, 4.9, "Session 1\nREST Baseline", ha='center', va='center', fontsize=8, color='white')
ax.text(1.65, 4.55, f"{sess_counts['r01']:,} windows", ha='center', va='center', fontsize=8, color='white')

r02_box = FancyBboxPatch((3.1, 4.4), 2.1, 1.5,
    boxstyle="round,pad=0.1", facecolor=C_R02, edgecolor='white', linewidth=1.5, alpha=0.85)
ax.add_patch(r02_box)
ax.text(4.15, 5.35, "r02", ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(4.15, 4.9, "Session 2\nREST Baseline", ha='center', va='center', fontsize=8, color='white')
ax.text(4.15, 4.55, f"{sess_counts['r02']:,} windows", ha='center', va='center', fontsize=8, color='white')

# Stimulus tags under r01/r02
ax.text(2.9, 4.0, "Stimuli: REST + Eyes Open/Closed\n(same modality, different weeks)",
        ha='center', va='center', fontsize=8, color='#1A5276',
        bbox=dict(boxstyle='round', facecolor='#D6EAF8', alpha=0.7))

# State labels
ax.text(1.65, 3.5, "State = 0", ha='center', fontsize=9, color=C_R01,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_R01, linewidth=1.2))
ax.text(4.15, 3.5, "State = 1", ha='center', fontsize=9, color=C_R02,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=C_R02, linewidth=1.2))

# ── Model box (center) ───────────────────────────────
model_box = FancyBboxPatch((5.9, 3.0), 2.2, 3.5,
    boxstyle="round,pad=0.15", facecolor='#F9F9F9', edgecolor='#555', linewidth=2)
ax.add_patch(model_box)
ax.text(7.0, 6.2, "DOMCS-EEG\nModel", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(7.0, 5.6, "CNN Encoder\n↓\nIdentity Branch\n(L2-norm embeds)\n↓\nKMeans Prototypes",
        ha='center', va='center', fontsize=8.5, color='#333',
        bbox=dict(boxstyle='round', facecolor='#EFEFEF', alpha=0.7))
ax.text(7.0, 3.3, "State Branch\n(Orthogonal,\ndetached)",
        ha='center', va='center', fontsize=8, color='#888')

# ── Probe box (right) ────────────────────────────────
probe_box = FancyBboxPatch((8.5, 3.2), 5.0, 3.1,
    boxstyle="round,pad=0.1", facecolor='#FDEDEC', edgecolor=C_TASK, linewidth=2.5)
ax.add_patch(probe_box)
ax.text(11.0, 6.0, "PROBE SET", ha='center', fontsize=12, fontweight='bold', color=C_TASK)

r03_box = FancyBboxPatch((9.5, 4.3), 3.0, 1.6,
    boxstyle="round,pad=0.1", facecolor=C_R03, edgecolor='white', linewidth=1.5, alpha=0.85)
ax.add_patch(r03_box)
ax.text(11.0, 5.35, "r03", ha='center', va='center', fontsize=13, fontweight='bold', color='white')
ax.text(11.0, 4.9, "Session 3 — TASK\nImage Viewing + SSVEP", ha='center', va='center',
        fontsize=8.5, color='white')
ax.text(11.0, 4.5, f"{sess_counts['r03']:,} windows (HELD OUT)", ha='center', va='center',
        fontsize=8, color='white')

ax.text(11.0, 4.0, "Stimuli: GAPED/OASIS images\n+ SSVEP (3/5/7/10 Hz)\n(NEW modality, never seen in training)",
        ha='center', va='center', fontsize=8, color='#922B21',
        bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8))

# ── Score box ────────────────────────────────────────
score_box = FancyBboxPatch((9.0, 2.3), 4.0, 0.75,
    boxstyle="round,pad=0.1", facecolor='#EAFAF1', edgecolor='#27AE60', linewidth=1.5)
ax.add_patch(score_box)
ax.text(11.0, 2.65, "EER = 24.07%  |  AUC = 0.836",
        ha='center', va='center', fontsize=11, fontweight='bold', color='#145A32')

# ── Arrows ───────────────────────────────────────────
# Enroll → model
ax.annotate('', xy=(5.9, 4.85), xytext=(5.5, 4.85),
            arrowprops=dict(arrowstyle='->', color=C_REST, lw=2.5))
# Model → probe
ax.annotate('', xy=(8.5, 4.85), xytext=(8.1, 4.85),
            arrowprops=dict(arrowstyle='->', color=C_TASK, lw=2.5))
ax.text(6.6, 5.05, "Train\n(identity\n+ state loss)", ha='center', fontsize=7.5, color=C_REST)
ax.text(8.35, 5.05, "Verify\n(cosine\nsimilarity)", ha='center', fontsize=7.5, color=C_TASK)

# Model → score
ax.annotate('', xy=(11.0, 3.07), xytext=(11.0, 3.2),
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2))

# ── Legend ────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(color=C_REST, label='REST (r01+r02): Enrollment'),
    mpatches.Patch(color=C_TASK, label='TASK (r03): Probe — strictly held out'),
    mpatches.Patch(color='#27AE60', label='Result (5-seed average)'),
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=9, framealpha=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG3_protocol_diagram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG3_protocol_diagram.png")

# ─────────────────────────────────────────────────────────────────
# FIG 4: Raw EEG sample — REST vs TASK (1 subject)
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 4 — Raw EEG samples …")

# Pick subject with data in all 3 sessions
target_sub = None
for sid in subjects:
    mask = y == sid
    has_r01 = ((sessions == 'r01') & mask).sum() > 0
    has_r02 = ((sessions == 'r02') & mask).sum() > 0
    has_r03 = ((sessions == 'r03') & mask).sum() > 0
    if has_r01 and has_r02 and has_r03:
        target_sub = sid
        break

rest_mask = (y == target_sub) & (sessions == 'r01')
task_mask = (y == target_sub) & (sessions == 'r03')

X_rest_sub = X[rest_mask][:3]  # 3 windows
X_task_sub = X[task_mask][:3]  # 3 windows

t = np.linspace(0, WIN_SEC, WIN_SAMPLES)
# Show 4 channels
ch_show = [0, 2, 5, 8]
ch_labels_show = [channels[c] if c < len(channels) else f'Ch{c+1}' for c in ch_show]

fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
fig.suptitle(f"Sample EEG Windows — Subject {target_sub}\n"
             f"Left: r01 (REST, 'stimuli rest') | Right: r03 (TASK, Image Viewing)",
             fontsize=12, fontweight='bold')

offset_scale = 100  # µV offset between windows

for row, (ch, ch_lbl) in enumerate(zip(ch_show, ch_labels_show)):
    for col, (X_sub, sess_lbl, col_color, stim_title) in enumerate([
        (X_rest_sub, 'r01 — REST', C_R01, 'stimuli: rest'),
        (X_task_sub, 'r03 — TASK', C_R03, 'stimuli: image viewing')
    ]):
        ax = axes[row, col]
        for w_idx in range(min(3, len(X_sub))):
            signal = X_sub[w_idx, ch, :]
            # Z-score for display
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            ax.plot(t, signal + w_idx * 3, color=col_color,
                    alpha=0.8 - w_idx * 0.15, linewidth=0.9,
                    label=f'Window {w_idx+1}' if row == 0 else None)
        ax.set_ylabel(ch_lbl, fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if row == 0:
            ax.set_title(f'{sess_lbl}\n({stim_title})', fontsize=10,
                         color=C_REST if col == 0 else C_TASK, fontweight='bold')
        if row == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_yticks([])
        ax.tick_params(labelsize=8)

# Add window legend on top-right panel
axes[0, 1].legend(loc='upper right', fontsize=8, framealpha=0.7)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG4_eeg_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG4_eeg_samples.png")

# ─────────────────────────────────────────────────────────────────
# FIG 5: Power Spectral Density — REST vs TASK (all subjects)
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 5 — PSD comparison …")

rest_mask_all = (sessions == 'r01') | (sessions == 'r02')
task_mask_all = (sessions == 'r03')

# Compute average PSD across all windows and channels
def mean_psd(X_sub, fs=128, nperseg=128):
    psds = []
    for win in X_sub:                   # (14, 256)
        for ch_data in win:             # (256,)
            f, psd = welch(ch_data.astype(np.float64), fs=fs, nperseg=nperseg)
            psds.append(psd)
    return f, np.mean(psds, axis=0)

# Sample up to 500 windows for speed
np.random.seed(42)
rest_idx = np.where(rest_mask_all)[0]
task_idx = np.where(task_mask_all)[0]
rest_sample = X[np.random.choice(rest_idx, min(500, len(rest_idx)), replace=False)]
task_sample = X[np.random.choice(task_idx, min(500, len(task_idx)), replace=False)]

f_rest, psd_rest = mean_psd(rest_sample)
f_task, psd_task = mean_psd(task_sample)

# Per-channel PSD for error bands
def per_channel_psd(X_sub, fs=128, nperseg=128):
    ch_psds = [[] for _ in range(X_sub.shape[1])]
    for win in X_sub:
        for ci, ch_data in enumerate(win):
            f, psd = welch(ch_data.astype(np.float64), fs=fs, nperseg=nperseg)
            ch_psds[ci].append(psd)
    return f, [np.mean(cp, axis=0) for cp in ch_psds]

f, rest_ch_psds = per_channel_psd(rest_sample)
_, task_ch_psds = per_channel_psd(task_sample)
rest_mean = np.mean(rest_ch_psds, axis=0)
rest_std  = np.std(rest_ch_psds, axis=0)
task_mean = np.mean(task_ch_psds, axis=0)
task_std  = np.std(task_ch_psds, axis=0)

# EEG band ranges
bands = {
    'δ (0.5–4 Hz)':   (0.5, 4),
    'θ (4–8 Hz)':     (4, 8),
    'α (8–13 Hz)':    (8, 13),
    'β (13–30 Hz)':   (13, 30),
    'γ (30–45 Hz)':   (30, 45),
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Power Spectral Density — BED Dataset\nREST (r01+r02) vs TASK (r03)",
             fontsize=13, fontweight='bold')

for ax, (title, mask_b, mean_b, std_b, color) in zip(axes, [
    ('REST sessions (r01 + r02)', rest_mask_all, rest_mean, rest_std, C_REST),
    ('TASK session (r03 — Image + SSVEP)', task_mask_all, task_mean, task_std, C_TASK),
]):
    freq_mask = f <= 50
    ax.semilogy(f[freq_mask], mean_b[freq_mask], color=color, linewidth=2, label='Mean PSD')
    ax.fill_between(f[freq_mask],
                    np.maximum(mean_b[freq_mask] - std_b[freq_mask], 1e-10),
                    mean_b[freq_mask] + std_b[freq_mask],
                    alpha=0.25, color=color, label='±1 SD (across channels)')

    # Band shading
    band_colors = ['#AED6F1', '#A9DFBF', '#FAD7A0', '#F5B7B1', '#D7BDE2']
    for (bname, (blo, bhi)), bc in zip(bands.items(), band_colors):
        ax.axvspan(blo, bhi, alpha=0.10, color=bc)
        ax.text((blo+bhi)/2, ax.get_ylim()[0] if False else 1e-3,
                bname.split(' ')[0], ha='center', fontsize=7.5, color='#555')

    # SSVEP peak markers for r03
    if 'SSVEP' in title:
        for freq_hz in [3, 5, 7, 10]:
            idx = np.argmin(np.abs(f - freq_hz))
            ax.axvline(freq_hz, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.text(freq_hz + 0.2, mean_b[idx] * 3, f'{freq_hz}Hz\nSSVEP',
                    fontsize=7, color='purple', va='bottom')

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold',
                 color=C_REST if 'REST' in title else C_TASK)
    ax.set_xlim(0.5, 50)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG5_psd_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG5_psd_comparison.png")

# ─────────────────────────────────────────────────────────────────
# FIG 6: Per-subject window balance across sessions
# ─────────────────────────────────────────────────────────────────
print("Generating Fig 6 — Per-subject balance …")
sub_counts = {sid: {} for sid in subjects}
for sid in subjects:
    mask = y == sid
    for sess in ['r01','r02','r03']:
        sub_counts[sid][sess] = int(((sessions == sess) & mask).sum())

sub_ids  = list(subjects)
r01_vals = [sub_counts[s]['r01'] for s in sub_ids]
r02_vals = [sub_counts[s]['r02'] for s in sub_ids]
r03_vals = [sub_counts[s]['r03'] for s in sub_ids]
x_pos    = np.arange(len(sub_ids))

fig, ax = plt.subplots(figsize=(16, 5))
w = 0.28
b1 = ax.bar(x_pos - w, r01_vals, w, label='r01 (REST, state=0)', color=C_R01, alpha=0.85)
b2 = ax.bar(x_pos,      r02_vals, w, label='r02 (REST, state=1)', color=C_R02, alpha=0.85)
b3 = ax.bar(x_pos + w, r03_vals, w, label='r03 (TASK, PROBE)',   color=C_R03, alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'S{s}' for s in sub_ids], fontsize=8, rotation=45)
ax.set_ylabel('Number of Windows', fontsize=11)
ax.set_xlabel('Subject ID', fontsize=11)
ax.set_title('Per-Subject Window Count per Session\n'
             '(Enroll: r01+r02 REST | Probe: r03 TASK)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add mean lines
ax.axhline(np.mean(r01_vals), color=C_R01, linestyle='--', linewidth=1.2, alpha=0.5)
ax.axhline(np.mean(r02_vals), color=C_R02, linestyle='--', linewidth=1.2, alpha=0.5)
ax.axhline(np.mean(r03_vals), color=C_R03, linestyle='--', linewidth=1.2, alpha=0.5)

plt.tight_layout()
fig.savefig(OUT_DIR / 'FIG6_per_subject_balance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved FIG6_per_subject_balance.png")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL FIGURES SAVED TO:", OUT_DIR)
print("="*60)
for f_name in sorted(OUT_DIR.glob('*.png')):
    size_kb = f_name.stat().st_size / 1024
    print(f"  {f_name.name:<40} {size_kb:.0f} KB")

print("\nInterpretation summary for paper:")
print(f"  r01: {sess_counts['r01']:,} REST windows (stimuli: rest + eyes open/closed)")
print(f"  r02: {sess_counts['r02']:,} REST windows (same types, session 2 — weeks later)")
print(f"  r03: {sess_counts['r03']:,} TASK windows (stimuli: image viewing GAPED/OASIS + SSVEP)")
print(f"  State labels in training: r01=0 (cross-session), r02=1 (cross-session)")
print(f"  Test: r03 is strictly held-out TASK probe set (never seen during training)")
print(f"  Challenge: r03 stimuli modality is completely different from r01/r02")
