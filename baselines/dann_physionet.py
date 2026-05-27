"""
DANN Baseline — PHYSIONET B2T Protocol (Table S8)
=================================================
Fresh DANN implementation for Table S8 comparison against DOMCS-EEG.

Architecture:
  - Same 3-layer CNN backbone as DOMCS-EEG (64ch, locked version)
  - Identity classifier head (ArcFace)
  - Domain adversarial head (gradient reversal + domain classifier)
    Domain = state (rest=0, task=1) — same definition as DOMCS-EEG

Protocol:
  - Same B2T data split as locked version
  - Enroll: R01+R02 (REST), Probe: R03-R14 (TASK)
  - Training: ALL windows with domain adversarial loss

5 seeds — results go into Table S8.

Run on Brev:
  cd /home/nvidia/24PHD1237/DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/code
  python /path/to/dann_physionet.py
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

# ─────────────────────────────────────────────────────────────
# 0. ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="DANN Baseline — PHYSIONET B2T Protocol (Table S8)"
    )
    p.add_argument(
        "--data", type=Path,
        default=Path("./data/preprocessed_b2t.npz"),
        help="Path to preprocessed PHYSIONET NPZ file (must contain X, y, session keys)"
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("./DANN_PHYSIONET_RESULTS"),
        help="Output directory for per-seed checkpoints and results JSON"
    )
    return p.parse_args()

_args    = _parse_args()
DATA_PATH = _args.data
OUT_DIR   = _args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────

# Shared with locked version
N_CHANNELS  = 64
WIN_SAMPLES = 256
ID_DIM      = 128
N_SUBJECTS  = 109
ARC_S       = 32.0
ARC_M       = 0.50

# DANN-specific
LAMBDA_DANN  = 1.0      # weight of domain adversarial loss
LAMBDA_ALPHA = 0.5      # GRL reversal strength (annealed during training)
EPOCHS       = 60
BATCH_SIZE   = 128
LR           = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.10
PATIENCE     = 12
KMEANS_K     = 3

REST_RUNS = ["R01", "R02"]    # enrollment sessions
TASK_RUNS = [f"R{i:02d}" for i in range(3, 15)]   # R03-R14

SEEDS = [1, 2, 3, 4, 5]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# 1. GRADIENT REVERSAL LAYER
# ─────────────────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha):
        self.alpha = alpha


# ─────────────────────────────────────────────────────────────
# 2. DANN MODEL
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


class DANNModel(nn.Module):
    """
    Domain Adversarial Neural Network for EEG biometrics.
    Encoder is identical to DOMCS-EEG locked version.
    Domain = state (rest vs task).
    """
    def __init__(self, n_subjects=109, n_channels=64, id_dim=128,
                 arc_s=32.0, arc_m=0.50):
        super().__init__()

        # Encoder — exactly matches DOMCS-EEG locked encoder
        self.encoder = nn.Sequential(
            ConvBlock(n_channels, 64,  7, 3),
            ConvBlock(64,         128, 5, 2),
            ConvBlock(128,        256, 3, 1),
            nn.AdaptiveAvgPool1d(1),
        )

        # Identity embedding head
        self.id_head = nn.Sequential(
            nn.Linear(256, id_dim, bias=False),
            nn.LayerNorm(id_dim)
        )

        # ArcFace weight
        self.arc_w = nn.Parameter(torch.FloatTensor(n_subjects, id_dim))
        nn.init.xavier_uniform_(self.arc_w)
        self.arc_s = arc_s
        self.arc_m = arc_m

        # Gradient reversal + domain classifier (binary: rest / task)
        self.grl         = GradientReversal(alpha=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)   # 2 domains: rest / task
        )

    def forward(self, x, domain_alpha=1.0):
        f = self.encoder(x).squeeze(-1)           # (B, 256)

        z_id     = F.normalize(self.id_head(f), dim=1)   # (B, id_dim)

        self.grl.set_alpha(domain_alpha)
        d_logit  = self.domain_head(self.grl(f))         # (B, 2)

        return z_id, d_logit, f

    def arcface_logits(self, z_id, labels):
        cos   = F.linear(F.normalize(z_id), F.normalize(self.arc_w))
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1,1), 1.0)
        return torch.cos(theta + self.arc_m * one_hot) * self.arc_s


# ─────────────────────────────────────────────────────────────
# 3. DATASET
# ─────────────────────────────────────────────────────────────

class EEGDataset(Dataset):
    def __init__(self, X, y, domain):
        self.X      = torch.tensor(X,      dtype=torch.float32)
        self.y      = torch.tensor(y,      dtype=torch.long)
        self.domain = torch.tensor(domain, dtype=torch.long)

    def __len__(self):  return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.domain[i]


# ─────────────────────────────────────────────────────────────
# 4. LOAD PHYSIONET B2T DATA (same split as locked version)
# ─────────────────────────────────────────────────────────────

def load_physionet_b2t(seed):
    """
    Load preprocessed PHYSIONET windows from the path given by --data.
    NPZ must contain keys: X (N,64,256 float32), y (N,) int64, session (N,) str.
    Session values must be uppercase, e.g. 'R01', 'R02', ..., 'R14'.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Data file not found: {DATA_PATH}\n"
            f"Pass the correct path with: --data /path/to/preprocessed_b2t.npz"
        )

    print(f"  Loading: {DATA_PATH}")
    data  = np.load(DATA_PATH, allow_pickle=True)
    X     = data["X"].astype(np.float32)
    y     = data["y"].astype(np.int64)
    runs  = np.asarray([str(r) for r in data["session"]], dtype=object)

    print(f"  Loaded: {X.shape}, subjects={len(np.unique(y))}, runs={sorted(set(runs))[:6]}...")

    assert X.shape[1] == 64,  f"Expected 64 channels, got {X.shape[1]}"
    assert X.shape[2] == 256, f"Expected 256 samples, got {X.shape[2]}"

    # Domain labels: rest=0, task=1
    domain = np.zeros(len(X), dtype=np.int64)
    for run in TASK_RUNS:
        domain[runs == run] = 1

    # Enrollment (REST) indices
    rest_idx = np.where(np.isin(runs, REST_RUNS))[0]
    task_idx = np.where(np.isin(runs, TASK_RUNS))[0]

    X_rest = X[rest_idx]; y_rest = y[rest_idx]; d_rest = domain[rest_idx]
    X_task = X[task_idx]; y_task = y[task_idx]; d_task = domain[task_idx]

    # B2T-safe: combine all for training (rest for identity, all for domain)
    # Val split from REST windows only (strict B2T)
    n_rest = len(X_rest)
    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(n_rest)
    n_val  = int(n_rest * VAL_SPLIT)

    val_rest_idx  = perm[:n_val]
    train_rest_idx = perm[n_val:]

    print(f"  REST train: {len(train_rest_idx)}, REST val: {len(val_rest_idx)}")
    print(f"  TASK windows (domain training only): {len(X_task)}")
    print(f"  Probe (test) = TASK windows (same as domain=1 but separate eval)")

    return {
        "X_rest_train": X_rest[train_rest_idx],
        "y_rest_train": y_rest[train_rest_idx],
        "d_rest_train": d_rest[train_rest_idx],
        "X_rest_val":   X_rest[val_rest_idx],
        "y_rest_val":   y_rest[val_rest_idx],
        "d_rest_val":   d_rest[val_rest_idx],
        "X_task":  X_task, "y_task": y_task, "d_task": d_task,
        "X_enroll": X_rest, "y_enroll": y_rest,  # full rest for enrollment
        "X_probe":  X_task, "y_probe":  y_task,  # full task as probe
        "n_subjects": len(np.unique(y)),
    }


# ─────────────────────────────────────────────────────────────
# 5. TRAIN ONE SEED
# ─────────────────────────────────────────────────────────────

def annealed_alpha(epoch, total_epochs, gamma=10.0):
    """GRL alpha schedule: 0 -> 1 as training progresses."""
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one_seed(seed, split):
    set_seed(seed)
    ckpt_dir = OUT_DIR / f"seed_{seed}"
    ckpt_dir.mkdir(exist_ok=True)

    n_subj = split["n_subjects"]
    model  = DANNModel(n_subj, N_CHANNELS, ID_DIM, ARC_S, ARC_M).to(DEVICE)

    # Training datasets
    # For identity loss: REST train only (B2T safe)
    # For domain loss  : REST train + TASK windows (all domains)
    rest_ds = EEGDataset(split["X_rest_train"], split["y_rest_train"], split["d_rest_train"])
    task_ds = EEGDataset(split["X_task"],       split["y_task"],       split["d_task"])
    val_ds  = EEGDataset(split["X_rest_val"],   split["y_rest_val"],   split["d_rest_val"])

    rest_loader = DataLoader(rest_ds, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    task_loader = DataLoader(task_ds, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val = float("inf"); best_epoch = 0; patience = 0; t0 = time.time()

    task_iter = iter(task_loader)   # cycling iterator over task windows

    for ep in range(1, EPOCHS + 1):
        alpha = annealed_alpha(ep, EPOCHS)
        model.train()
        tr_loss = 0.0

        for xb_rest, yb_rest, db_rest in rest_loader:
            xb_rest  = xb_rest.to(DEVICE, non_blocking=True)
            yb_rest  = yb_rest.to(DEVICE, non_blocking=True)
            db_rest  = db_rest.to(DEVICE, non_blocking=True)

            # Get a task batch for domain loss
            try:
                xb_task, _, db_task = next(task_iter)
            except StopIteration:
                task_iter = iter(task_loader)
                xb_task, _, db_task = next(task_iter)

            xb_task = xb_task.to(DEVICE, non_blocking=True)
            db_task = db_task.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                # Identity loss on REST only (B2T safe)
                z_id, d_logit_rest, _ = model(xb_rest, domain_alpha=alpha)
                l_arc = F.cross_entropy(model.arcface_logits(z_id, yb_rest), yb_rest)

                # Domain loss on REST + TASK
                l_domain_rest = F.cross_entropy(d_logit_rest, db_rest)
                _, d_logit_task, _ = model(xb_task, domain_alpha=alpha)
                l_domain_task = F.cross_entropy(d_logit_task, db_task)
                l_domain = 0.5 * (l_domain_rest + l_domain_task)

                loss = l_arc + LAMBDA_DANN * l_domain

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        tr_loss /= len(rest_loader)

        # Val loss (REST val only)
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb, db in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE); db = db.to(DEVICE)
                z_id, d_logit, _ = model(xb, domain_alpha=alpha)
                l_arc    = F.cross_entropy(model.arcface_logits(z_id, yb), yb)
                l_domain = F.cross_entropy(d_logit, db)
                va_loss += (l_arc + LAMBDA_DANN * l_domain).item()
        va_loss /= len(val_loader)

        improved = va_loss < best_val
        if improved:
            best_val = va_loss; best_epoch = ep; patience = 0
            torch.save({"model_state": model.state_dict(), "epoch": ep,
                        "seed": seed, "n_subjects": n_subj}, ckpt_dir / "model_best.pt")
        else:
            patience += 1

        flag = "*" if improved else ""
        print(f"  ep {ep:03d} train={tr_loss:.4f} val={va_loss:.4f} alpha={alpha:.3f} {flag}")

        if patience >= PATIENCE:
            print(f"  Early stop at epoch {ep}"); break

    # Evaluate
    ckpt = torch.load(ckpt_dir / "model_best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eer_pct, auc_val, crr_pct = evaluate(
        model,
        split["X_enroll"], split["y_enroll"],
        split["X_probe"],  split["y_probe"]
    )
    train_min = (time.time() - t0) / 60
    print(f"  SEED {seed}: EER={eer_pct:.4f}%  AUC={auc_val:.4f}  CRR={crr_pct:.2f}%  ({train_min:.1f} min)")

    return {"seed": seed, "best_epoch": best_epoch, "train_min": train_min,
            "eer_percent": eer_pct, "auc": auc_val, "crr_percent": crr_pct}


# ─────────────────────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────────────────────

def extract_embeddings(model, X_np, y_np):
    dummy_d = np.zeros(len(X_np), dtype=np.int64)
    ds = EEGDataset(X_np, y_np, dummy_d)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    E, Y = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(DEVICE)
            z_id, *_ = model(xb)
            E.append(z_id.cpu().numpy())
            Y.append(yb.numpy())
    E = np.concatenate(E); Y = np.concatenate(Y)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E.astype(np.float32), Y.astype(np.int64)


def build_prototypes(E, Y):
    pvecs, powner = [], []
    for sid in sorted(np.unique(Y)):
        idx = np.where(Y == sid)[0]
        k_use = min(KMEANS_K, len(idx))
        km = KMeans(n_clusters=k_use, random_state=42, n_init=10)
        km.fit(E[idx])
        for c in km.cluster_centers_:
            c = c / (np.linalg.norm(c) + 1e-12)
            pvecs.append(c.astype(np.float32)); powner.append(int(sid))
    return np.stack(pvecs), np.asarray(powner, dtype=np.int64)


def evaluate(model, X_enroll, y_enroll, X_probe, y_probe):
    E_en, Y_en = extract_embeddings(model, X_enroll, y_enroll)
    E_pr, Y_pr = extract_embeddings(model, X_probe,  y_probe)
    P, P_owner = build_prototypes(E_en, Y_en)

    scores, labels = [], []
    for i in range(len(E_pr)):
        e = E_pr[i]; tid = Y_pr[i]
        sim = e @ P.T
        scores.append(float(sim[P_owner == tid].max())); labels.append(1)
        for sid in np.unique(P_owner):
            if sid == tid: continue
            scores.append(float(sim[P_owner == sid].max())); labels.append(0)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc_val = sk_auc(fpr, tpr)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0

    sim_all  = E_pr @ P.T
    subj_ids = sorted(np.unique(Y_en))
    sc = np.zeros((len(E_pr), len(subj_ids)))
    for j, sid in enumerate(subj_ids):
        sc[:, j] = sim_all[:, P_owner == sid].max(axis=1)
    pred = np.array([subj_ids[j] for j in sc.argmax(axis=1)])
    crr  = float(np.mean(pred == Y_pr)) * 100

    return float(eer * 100), float(auc_val), crr


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DANN BASELINE — PHYSIONET B2T PROTOCOL (Table S8)")
    print("=" * 60)

    all_results = []

    for seed in SEEDS:
        print(f"\n{'─'*50}")
        print(f"  SEED {seed} / {SEEDS[-1]}")

        split = load_physionet_b2t(seed)
        res   = train_one_seed(seed, split)
        all_results.append(res)

    eers = [r["eer_percent"] for r in all_results]
    aucs = [r["auc"]         for r in all_results]
    crrs = [r["crr_percent"] for r in all_results]

    summary = {
        "model":    "DANN",
        "dataset":  "PHYSIONET",
        "protocol": "B2T (R01+R02 enroll, R03-R14 probe)",
        "seeds":    SEEDS,
        "per_seed": all_results,
        "eer_mean": float(np.mean(eers)),
        "eer_std":  float(np.std(eers, ddof=1)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs, ddof=1)),
        "crr_mean": float(np.mean(crrs)),
        "hyperparams": {
            "lambda_dann": LAMBDA_DANN,
            "arc_s": ARC_S, "arc_m": ARC_M,
            "val_split": VAL_SPLIT,
        }
    }

    with open(OUT_DIR / "DANN_PHYSIONET_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  DANN FINAL RESULTS — TABLE S8")
    print("=" * 60)
    for r in all_results:
        print(f"  Seed {r['seed']}: EER={r['eer_percent']:.4f}%  AUC={r['auc']:.4f}  CRR={r['crr_percent']:.2f}%")
    print(f"\n  EER: {np.mean(eers):.4f}% ± {np.std(eers, ddof=1):.4f}%")
    print(f"  AUC: {np.mean(aucs):.4f}")
    print(f"\n  Results saved: {OUT_DIR}/DANN_PHYSIONET_SUMMARY.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
