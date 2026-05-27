"""
config.py — DOMCS-EEG Final Locked Implementation
All hyperparameters and paths in one place.
Auto-detects Brev (NVIDIA) vs local environment.
"""
import os
from pathlib import Path

# ─── Environment detection ────────────────────────────────────────────────────
def _is_brev():
    return os.path.exists("/home/nvidia/24PHD1237")

# ─── Root paths ──────────────────────────────────────────────────────────────
if _is_brev():
    DATA_ROOT  = Path("/home/nvidia/24PHD1237/FILES_1/EEGMMIDB")
    EXP_ROOT   = Path("/home/nvidia/24PHD1237/DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION")
else:
    # Local fallback — update if your local path differs
    DATA_ROOT  = Path(r"C:\Users\L.KANIMOZHI\OneDrive\23may26 rescinded email paper1\data")
    EXP_ROOT   = Path(r"C:\Users\L.KANIMOZHI\OneDrive\23may26 rescinded email paper1\DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION")

# ─── Data paths ───────────────────────────────────────────────────────────────
NPZ_PATH         = DATA_ROOT / "EEGMMIDB_win2s_step1s_fs128.npz"
NPZ_EVENT_AWARE  = DATA_ROOT / "EEGMMIDB_event_aware_fs128.npz"       # Approach②
NPZ_APPROACH3    = DATA_ROOT / "EEGMMIDB_approach3_T0rest_fs128.npz"  # Approach③

# ─── Output directories ───────────────────────────────────────────────────────
CODE_DIR        = EXP_ROOT / "code"
CKPT_DIR        = EXP_ROOT / "checkpoints"
LOG_DIR         = EXP_ROOT / "logs"
FIG_DIR         = EXP_ROOT / "figures"
TABLE_DIR       = EXP_ROOT / "tables"
LATEX_DIR       = EXP_ROOT / "latex_tables"
PNG_DIR         = EXP_ROOT / "paper_figures_png"
PDF_DIR         = EXP_ROOT / "paper_figures_pdf"
VERIF_DIR       = EXP_ROOT / "verification"
SUPP_DIR        = EXP_ROOT / "supplementary"

# Create all output directories
for _d in [CKPT_DIR, LOG_DIR, FIG_DIR, TABLE_DIR, LATEX_DIR,
           PNG_DIR, PDF_DIR, VERIF_DIR, SUPP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Data constants ───────────────────────────────────────────────────────────
N_SUBJECTS     = 109
N_CHANNELS     = 64
WIN_SAMPLES    = 256          # 2s @ 128Hz
FS             = 128

REST_RUNS      = ['R01', 'R02']
TASK_RUNS      = ['R03', 'R04', 'R05', 'R06', 'R07', 'R08',
                  'R09', 'R10', 'R11', 'R12', 'R13', 'R14']

# State label mapping
STATE_REST = 0
STATE_TASK = 1

# Task group labels (for fine-grained state labels)
TASK_GROUP_MAP = {
    'R03': 1, 'R07': 1, 'R11': 1,   # RL-Fist MI
    'R04': 2, 'R08': 2, 'R12': 2,   # BF-Feet MI
    'R05': 3, 'R09': 3, 'R13': 3,   # RL-Fist MV
    'R06': 4, 'R10': 4, 'R14': 4,   # BF-Feet MV
}
N_STATE_CLASSES = 2   # binary: rest vs task (change to 5 for fine-grained)

# ─── Architecture hyperparameters ─────────────────────────────────────────────
ENCODER_CHANNELS = [64, 64, 128, 256]   # input + 3 conv layers
ID_DIM    = 128    # identity embedding dimension
STATE_DIM = 128    # state embedding dimension

# ArcFace
ARC_S     = 32.0
ARC_M     = 0.50

# ─── Training hyperparameters ─────────────────────────────────────────────────
SEEDS          = [1, 2, 3, 4, 5]
N_EPOCHS       = 60
BATCH_SIZE     = 256
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
VAL_SPLIT      = 0.10    # fraction of windows held out for validation

# Loss weights
LAMBDA_SUPCON  = 0.30
LAMBDA_STATE   = 0.50
LAMBDA_ORTH    = 0.10

# Cosine LR scheduler
LR_MIN         = 1e-6

# ─── Gallery (enrollment) settings ────────────────────────────────────────────
GALLERY_K      = 3       # KMeans clusters per subject
GALLERY_RUNS   = REST_RUNS   # ALWAYS only rest runs for enrollment

# ─── Security evaluation ──────────────────────────────────────────────────────
FGSM_EPS_LIST  = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010]
PGD_EPS_LIST   = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010]
PGD_STEPS      = 10
PGD_ALPHA      = 0.001

AWGN_SNR_LIST  = [30, 20, 10, 5]

# ─── Matplotlib style ─────────────────────────────────────────────────────────
MPL_STYLE = {
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "figure.dpi":     300,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
}

COLORS = {
    "clean":   "#1f77b4",
    "T0":      "#ff7f0e",
    "T1":      "#2ca02c",
    "fgsm":    "#d62728",
    "pgd":     "#9467bd",
    "awgn":    "#1f77b4",
    "z_id":    "#2ca02c",
    "z_state": "#9467bd",
}
