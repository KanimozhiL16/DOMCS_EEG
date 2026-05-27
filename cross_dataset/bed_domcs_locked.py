"""
BED Cross-Dataset Validation — DOMCS-EEG Locked Implementation
==============================================================
Architecture: EXACTLY matches DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION
  - 3-layer 1D CNN encoder (14->64->128->256)
  - IdentityBranch: Linear(256->128, bias=False) -> LayerNorm -> L2-norm
  - StateBranch:    f.detach() -> Linear(256->128, bias=False) -> LayerNorm -> L2-norm
  - ArcFace s=32.0, m=0.50
  - lambda_supcon=0.30, lambda_state=0.50, lambda_orth=0.10

Protocol (BED-B2T):
  - Enroll/train: r01 + r02  (cross-session: state labels s=0 / s=1)
  - Test/probe  : r03
  - Identity loss (ArcFace + SupCon): on r01+r02 training windows ONLY (B2T safe)
  - State + Orth loss: on ALL training windows

Run on Brev:
  cd /home/nvidia/24PHD1237/DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/code
  python /path/to/bed_domcs_locked.py

Prerequisites:
  BED_win2s_step1s_fs128.npz must be at /home/nvidia/24PHD1237/BED_DATASET/
  (use rclone to copy from Google Drive — see rclone_bed_setup.sh)
"""

import os, json, time, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# 0. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="BED Cross-Dataset Validation — DOMCS-EEG Locked Implementation"
    )
    p.add_argument(
        "--data", type=Path,
        default=Path("./data/BED_win2s_step1s_fs128.npz"),
        help="Path to BED NPZ file (BED_win2s_step1s_fs128.npz)"
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("./BED_DOMCS_LOCKED_RESULTS"),
        help="Output directory for per-seed checkpoints and results JSON"
    )
    return p.parse_args()

_args   = _parse_args()
BED_NPZ = _args.data
OUT_DIR = _args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Locked hyperparameters (match DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/config.py)
N_CHANNELS    = 14        # BED: 14 Emotiv EPOC+ channels
WIN_SAMPLES   = 256       # 2s @ 128 Hz
ID_DIM        = 128
STATE_DIM     = 128
ARC_S         = 32.0      # LOCKED (not 30)
ARC_M         = 0.50
LAMBDA_SUPCON = 0.30      # LOCKED (not 0.50)
LAMBDA_STATE  = 0.50
LAMBDA_ORTH   = 0.10
EPOCHS        = 60
BATCH_SIZE    = 128
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
VAL_SPLIT     = 0.10      # LOCKED (not 0.20)
TEMPERATURE   = 0.07
PATIENCE      = 12
KMEANS_K      = 3

TRAIN_RUNS    = ["r01", "r02"]
TEST_RUNS     = ["r03"]
SEEDS         = [1, 2, 3, 4, 5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    import subprocess
    gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"GPU: {gpu}")


# ─────────────────────────────────────────────────────────────
# 1. MODEL — exact locked architecture, N_CHANNELS=14
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, pad):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ELU()
        )
    def forward(self, x): return self.net(x)


class EEGEncoder(nn.Module):
    """3-layer 1D CNN — matches locked version exactly (14ch instead of 64ch)."""
    def __init__(self, n_channels=14):
        super().__init__()
        self.conv1 = ConvBlock(n_channels, 64,  7, 3)   # 14->64,  k=7
        self.conv2 = ConvBlock(64,         128, 5, 2)   # 64->128, k=5
        self.conv3 = ConvBlock(128,        256, 3, 1)   # 128->256,k=3
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x).squeeze(-1)   # (B, 256)


class IdentityBranch(nn.Module):
    """Single Linear + LayerNorm + L2-norm. NO BN, NO ReLU. Matches locked version."""
    def __init__(self, enc_dim=256, id_dim=128):
        super().__init__()
        self.fc   = nn.Linear(enc_dim, id_dim, bias=False)
        self.norm = nn.LayerNorm(id_dim)

    def forward(self, f):
        return F.normalize(self.norm(self.fc(f)), dim=1)


class StateBranch(nn.Module):
    """f.detach() barrier + Single Linear + LayerNorm + L2-norm. Matches locked version."""
    def __init__(self, enc_dim=256, state_dim=128):
        super().__init__()
        self.fc   = nn.Linear(enc_dim, state_dim, bias=False)
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, f):
        return F.normalize(self.norm(self.fc(f.detach())), dim=1)


class DOMCSEEGModel(nn.Module):
    def __init__(self, n_subjects, n_channels=14,
                 id_dim=128, state_dim=128,
                 arc_s=32.0, arc_m=0.50):
        super().__init__()
        self.encoder      = EEGEncoder(n_channels)
        self.id_branch    = IdentityBranch(256, id_dim)
        self.state_branch = StateBranch(256, state_dim)

        # ArcFace weight matrix
        self.arc_w = nn.Parameter(torch.FloatTensor(n_subjects, id_dim))
        nn.init.xavier_uniform_(self.arc_w)
        self.arc_s = arc_s
        self.arc_m = arc_m

        # State classifier on z_state (correct — not z_id)
        self.state_cls = nn.Linear(state_dim, 2)

    def forward(self, x):
        f       = self.encoder(x)
        z_id    = self.id_branch(f)
        z_state = self.state_branch(f)          # detach inside branch
        s_logit = self.state_cls(z_state)
        return z_id, z_state, s_logit, f

    def arcface_logits(self, z_id, labels):
        cos   = F.linear(F.normalize(z_id), F.normalize(self.arc_w))
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1,1), 1.0)
        return torch.cos(theta + self.arc_m * one_hot) * self.arc_s


# ─────────────────────────────────────────────────────────────
# 2. LOSSES
# ─────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z, labels):
        z   = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.T
        lbl = labels.contiguous().view(-1, 1)
        mask = torch.eq(lbl, lbl.T).float().to(z.device)
        mask.fill_diagonal_(0)
        logits  = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp_log = torch.exp(logits)
        log_prob = logits - torch.log(exp_log.sum(dim=1, keepdim=True) + 1e-12)
        mean_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_pos.mean()


def orth_loss(z_id, z_state):
    dot = (F.normalize(z_id, dim=1) * F.normalize(z_state, dim=1)).sum(dim=1)
    return (dot ** 2).mean()


# ─────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X, y, state):
        self.X     = torch.tensor(X,     dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.long)
        self.state = torch.tensor(state, dtype=torch.long)

    def __len__(self):  return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.state[i]


# ─────────────────────────────────────────────────────────────
# 4. LOAD & SPLIT DATA
# ─────────────────────────────────────────────────────────────

def load_and_split(seed):
    data = np.load(BED_NPZ, allow_pickle=True)
    X    = data["X"].astype(np.float32)
    y    = data["y"].astype(np.int64)
    runs_raw = data["session"] if "session" in data.files else data["runs"]
    runs = np.asarray([str(r) for r in runs_raw], dtype=object)

    # Verify dataset
    assert X.shape[1] == 14,  f"Expected 14 channels, got {X.shape[1]}"
    assert X.shape[2] == 256, f"Expected 256 samples, got {X.shape[2]}"
    n_subj = len(np.unique(y))
    print(f"  BED: {X.shape}, {n_subj} subjects, runs={sorted(set(runs))}")

    # State labels per window: r01=0, r02=1 (cross-session state)
    state_labels = np.zeros(len(X), dtype=np.int64)
    state_labels[runs == "r02"] = 1
    state_labels[runs == "r03"] = 2   # test session (not used in training state loss)

    # Indices
    train_idx = np.where(np.isin(runs, TRAIN_RUNS))[0]
    test_idx  = np.where(np.isin(runs, TEST_RUNS))[0]

    X_tr_all = X[train_idx];  y_tr_all = y[train_idx]
    X_te     = X[test_idx];   y_te     = y[test_idx]
    s_tr_all = state_labels[train_idx]
    s_te     = state_labels[test_idx]

    # Validation split (10% — locked version)
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_tr_all)),
        test_size=VAL_SPLIT,
        stratify=y_tr_all,
        random_state=seed
    )

    print(f"  Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(X_te)}")
    return (X_tr_all, y_tr_all, s_tr_all,
            tr_idx, val_idx,
            X_te, y_te, s_te,
            n_subj)


# ─────────────────────────────────────────────────────────────
# 5. TRAIN ONE SEED
# ─────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def train_one_seed(seed, n_subj, X_tr_all, y_tr_all, s_tr_all,
                   tr_idx, val_idx, X_te, y_te, s_te):

    set_seed(seed)
    ckpt_dir = OUT_DIR / f"seed_{seed}"
    ckpt_dir.mkdir(exist_ok=True)

    # Datasets
    # B2T-safe: ALL training windows for state+orth (both r01 and r02)
    # Identity (ArcFace+SupCon): also on ALL training (r01+r02 — both are "rest-equivalent")
    train_ds = EEGDataset(X_tr_all[tr_idx], y_tr_all[tr_idx], s_tr_all[tr_idx])
    val_ds   = EEGDataset(X_tr_all[val_idx], y_tr_all[val_idx], s_tr_all[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = DOMCSEEGModel(n_subj, N_CHANNELS, ID_DIM, STATE_DIM, ARC_S, ARC_M).to(DEVICE)
    supcon = SupConLoss(TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val   = float("inf")
    best_epoch = 0
    patience   = 0
    logs       = []
    t0         = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for xb, yb, sb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            sb = sb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                z_id, z_state, s_logit, _ = model(xb)

                l_arc   = F.cross_entropy(model.arcface_logits(z_id, yb), yb)
                l_sc    = supcon(z_id, yb)
                l_state = F.cross_entropy(s_logit, sb)
                l_orth  = orth_loss(z_id, z_state)

                loss = l_arc + LAMBDA_SUPCON*l_sc + LAMBDA_STATE*l_state + LAMBDA_ORTH*l_orth

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # ── Val ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE); sb = sb.to(DEVICE)
                z_id, z_state, s_logit, _ = model(xb)
                l_arc   = F.cross_entropy(model.arcface_logits(z_id, yb), yb)
                l_sc    = supcon(z_id, yb)
                l_state = F.cross_entropy(s_logit, sb)
                l_orth  = orth_loss(z_id, z_state)
                loss    = l_arc + LAMBDA_SUPCON*l_sc + LAMBDA_STATE*l_state + LAMBDA_ORTH*l_orth
                va_loss += loss.item()
        va_loss /= len(val_loader)

        improved = va_loss < best_val
        if improved:
            best_val   = va_loss
            best_epoch = ep
            patience   = 0
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "best_val": best_val, "seed": seed,
                "n_subj": n_subj, "n_channels": N_CHANNELS,
                "id_dim": ID_DIM, "state_dim": STATE_DIM,
                "arc_s": ARC_S, "arc_m": ARC_M,
            }, ckpt_dir / "model_best.pt")
        else:
            patience += 1

        flag = "*" if improved else ""
        print(f"  ep {ep:03d} train={tr_loss:.4f} val={va_loss:.4f} {flag}")

        logs.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})

        if patience >= PATIENCE:
            print(f"  Early stop at epoch {ep}")
            break

    train_min = (time.time() - t0) / 60
    print(f"  Seed {seed} done. Best epoch={best_epoch}, train={train_min:.1f}min")

    # ── Load best and evaluate ──
    ckpt = torch.load(ckpt_dir / "model_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eer_pct, auc_val, crr_pct = evaluate(model, X_tr_all, y_tr_all, s_tr_all,
                                          X_te, y_te, s_te)

    print(f"  SEED {seed}: EER={eer_pct:.4f}% | AUC={auc_val:.4f} | CRR={crr_pct:.2f}%")

    return {
        "seed": seed, "best_epoch": best_epoch, "train_min": train_min,
        "eer_percent": eer_pct, "auc": auc_val, "crr_percent": crr_pct
    }


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION — KMeans enrollment prototypes
# ─────────────────────────────────────────────────────────────

def extract_embeddings(model, X_np, y_np, s_np):
    ds = EEGDataset(X_np, y_np, s_np)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    E, Y = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(DEVICE)
            z_id, *_ = model(xb)
            E.append(z_id.cpu().numpy())
            Y.append(yb.numpy())
    E = np.concatenate(E)
    Y = np.concatenate(Y)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32), Y.astype(np.int64)


def build_prototypes(E, Y, K=3):
    pvecs, powner = [], []
    for sid in sorted(np.unique(Y)):
        idx = np.where(Y == sid)[0]
        k_use = min(K, len(idx))
        km = KMeans(n_clusters=k_use, random_state=42, n_init=10)
        km.fit(E[idx])
        for c in km.cluster_centers_:
            c = c / (np.linalg.norm(c) + 1e-12)
            pvecs.append(c.astype(np.float32)); powner.append(int(sid))
    return np.stack(pvecs), np.asarray(powner, dtype=np.int64)


def evaluate(model, X_enroll, y_enroll, s_enroll, X_test, y_test, s_test):
    E_en, Y_en = extract_embeddings(model, X_enroll, y_enroll, s_enroll)
    E_te, Y_te = extract_embeddings(model, X_test,  y_test,  s_test)

    P, P_owner = build_prototypes(E_en, Y_en, K=KMEANS_K)

    scores, labels = [], []
    for i in range(len(E_te)):
        e = E_te[i]; tid = Y_te[i]
        sim = e @ P.T
        scores.append(float(sim[P_owner == tid].max())); labels.append(1)
        for sid in np.unique(P_owner):
            if sid == tid: continue
            scores.append(float(sim[P_owner == sid].max())); labels.append(0)

    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    auc_val = sk_auc(fpr, tpr)
    fnr     = 1.0 - tpr
    idx     = np.nanargmin(np.abs(fpr - fnr))
    eer     = (fpr[idx] + fnr[idx]) / 2.0

    # CRR
    sim_all = E_te @ P.T
    subj_ids = sorted(np.unique(Y_en))
    subj_score = np.zeros((len(E_te), len(subj_ids)))
    for j, sid in enumerate(subj_ids):
        subj_score[:, j] = sim_all[:, P_owner == sid].max(axis=1)
    pred = np.array([subj_ids[j] for j in subj_score.argmax(axis=1)])
    crr  = float(np.mean(pred == Y_te)) * 100

    return float(eer * 100), float(auc_val), crr


# ─────────────────────────────────────────────────────────────
# 7. MAIN — run 5 seeds
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BED DOMCS-EEG — LOCKED IMPLEMENTATION")
    print("=" * 60)

    if not BED_NPZ.exists():
        raise FileNotFoundError(
            f"\n[ERROR] BED dataset not found: {BED_NPZ}\n"
            f"Run rclone_bed_setup.sh first to copy from Google Drive."
        )

    # Load data once
    (X_tr_all, y_tr_all, s_tr_all,
     tr_idx_dummy, val_idx_dummy,
     X_te, y_te, s_te, n_subj) = load_and_split(seed=1)

    all_results = []

    for seed in SEEDS:
        print(f"\n{'─'*50}")
        print(f"  SEED {seed} / {SEEDS[-1]}")

        # Resplit per seed
        (X_tr_all, y_tr_all, s_tr_all,
         tr_idx, val_idx,
         X_te, y_te, s_te, n_subj) = load_and_split(seed)

        res = train_one_seed(seed, n_subj,
                             X_tr_all, y_tr_all, s_tr_all,
                             tr_idx, val_idx,
                             X_te, y_te, s_te)
        all_results.append(res)

    # ── Aggregate 5-seed statistics ──
    eers = [r["eer_percent"] for r in all_results]
    aucs = [r["auc"]         for r in all_results]
    crrs = [r["crr_percent"] for r in all_results]

    summary = {
        "model":    "DOMCS-EEG",
        "dataset":  "BED",
        "protocol": "BED-B2T (r01+r02 enroll, r03 probe)",
        "n_subjects": int(n_subj),
        "n_channels": N_CHANNELS,
        "seeds": SEEDS,
        "per_seed": all_results,
        "eer_mean":  float(np.mean(eers)),
        "eer_std":   float(np.std(eers, ddof=1)),
        "auc_mean":  float(np.mean(aucs)),
        "auc_std":   float(np.std(aucs, ddof=1)),
        "crr_mean":  float(np.mean(crrs)),
        "hyperparams": {
            "arc_s": ARC_S, "arc_m": ARC_M,
            "lambda_supcon": LAMBDA_SUPCON,
            "lambda_state":  LAMBDA_STATE,
            "lambda_orth":   LAMBDA_ORTH,
            "val_split": VAL_SPLIT,
        }
    }

    with open(OUT_DIR / "BED_DOMCS_LOCKED_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  BED DOMCS-EEG FINAL RESULTS (5 seeds)")
    print("=" * 60)
    for r in all_results:
        print(f"  Seed {r['seed']}: EER={r['eer_percent']:.4f}%  AUC={r['auc']:.4f}  CRR={r['crr_percent']:.2f}%")
    print(f"\n  EER: {np.mean(eers):.4f}% ± {np.std(eers, ddof=1):.4f}%")
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    print(f"  CRR: {np.mean(crrs):.2f}%")
    print(f"\n  Results saved: {OUT_DIR}/BED_DOMCS_LOCKED_SUMMARY.json")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
