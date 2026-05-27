"""
BED Baselines — EEGNet, DeepConvNet, CNN+ArcFace
=================================================
Same BED-B2T protocol as bed_domcs_locked.py:
  Enroll/train : r01 + r02
  Probe/test   : r03
  Eval         : KMeans prototypes + cosine similarity

All baselines use ArcFace (s=32, m=0.50) to match DOMCS-EEG locked version.
5 seeds each. Results go into Table S3 supplementary.

Run on Brev:
  cd /home/nvidia/24PHD1237/
  python bed_baselines.py 2>&1 | tee bed_baselines_log.txt
"""

import json, time, random, argparse
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
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(
        description="BED Baselines — EEGNet, DeepConvNet, CNN+ArcFace"
    )
    p.add_argument(
        "--data", type=Path,
        default=Path("./data/BED_win2s_step1s_fs128.npz"),
        help="Path to BED NPZ file (BED_win2s_step1s_fs128.npz)"
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("./BED_BASELINE_RESULTS"),
        help="Output directory for per-seed results"
    )
    return p.parse_args()

_args   = _parse_args()
BED_NPZ = _args.data
OUT_DIR = _args.out
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
N_CHANNELS  = 14
WIN_SAMPLES = 256
N_SUBJECTS  = 21
EMB_DIM     = 128
ARC_S       = 32.0     # locked version value
ARC_M       = 0.50
EPOCHS      = 60
BATCH_SIZE  = 128
LR          = 3e-4
WEIGHT_DECAY= 1e-4
VAL_SPLIT   = 0.10
PATIENCE    = 12
KMEANS_K    = 3
TRAIN_RUNS  = ["r01", "r02"]
TEST_RUNS   = ["r03"]
SEEDS       = [1, 2, 3, 4, 5]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─────────────────────────────────────────────────────────────
# ARCFACE
# ─────────────────────────────────────────────────────────────
class ArcFaceLayer(nn.Module):
    def __init__(self, emb_dim, n_cls, s=32.0, m=0.50):
        super().__init__()
        self.s = s; self.m = m
        self.W = nn.Parameter(torch.FloatTensor(n_cls, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, labels):
        cos   = F.linear(F.normalize(z), F.normalize(self.W))
        theta = torch.acos(cos.clamp(-1+1e-7, 1-1e-7))
        oh    = torch.zeros_like(cos).scatter_(1, labels.view(-1,1), 1.0)
        return torch.cos(theta + self.m * oh) * self.s


# ─────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────

class EEGNetBaseline(nn.Module):
    """EEGNet adapted for 14-channel BED input."""
    def __init__(self, Cin=14, T=256, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (Cin, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1,1,Cin,T)).flatten(1).shape[1]
        self.embed = nn.Sequential(nn.Linear(flat, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x):
        h = self.features(x.unsqueeze(1)).flatten(1)
        return F.normalize(self.embed(h), dim=1)


class DeepConvNetBaseline(nn.Module):
    """DeepConvNet adapted for 14-channel BED input."""
    def __init__(self, Cin=14, T=256, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1,5), bias=False),
            nn.Conv2d(25, 25, (Cin,1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d((1,2)), nn.Dropout(0.25),
            nn.Conv2d(25, 50, (1,5), bias=False),
            nn.BatchNorm2d(50), nn.ELU(),
            nn.MaxPool2d((1,2)), nn.Dropout(0.25),
            nn.Conv2d(50, 100, (1,5), bias=False),
            nn.BatchNorm2d(100), nn.ELU(),
            nn.MaxPool2d((1,2)), nn.Dropout(0.25),
            nn.Conv2d(100, 200, (1,5), bias=False),
            nn.BatchNorm2d(200), nn.ELU(),
            nn.MaxPool2d((1,2)), nn.Dropout(0.25),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1,1,Cin,T)).flatten(1).shape[1]
        self.embed = nn.Sequential(nn.Linear(flat, emb_dim), nn.LayerNorm(emb_dim))

    def forward(self, x):
        h = self.features(x.unsqueeze(1)).flatten(1)
        return F.normalize(self.embed(h), dim=1)


class CNNArcFaceBaseline(nn.Module):
    """3-layer 1D CNN + ArcFace (same backbone depth as DOMCS, no dual-space losses)."""
    def __init__(self, Cin=14, emb_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(Cin,  64,  7, padding=3, bias=False), nn.BatchNorm1d(64),  nn.ELU(),
            nn.Conv1d(64,  128,  5, padding=2, bias=False), nn.BatchNorm1d(128), nn.ELU(),
            nn.Conv1d(128, 256,  3, padding=1, bias=False), nn.BatchNorm1d(256), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Single Linear+LayerNorm head — same as DOMCS locked id_branch
        self.embed = nn.Sequential(
            nn.Linear(256, emb_dim, bias=False),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        h = self.backbone(x).squeeze(-1)
        return F.normalize(self.embed(h), dim=1)


MODELS = {
    "EEGNet":      lambda: EEGNetBaseline(N_CHANNELS, WIN_SAMPLES, EMB_DIM),
    "DeepConvNet": lambda: DeepConvNetBaseline(N_CHANNELS, WIN_SAMPLES, EMB_DIM),
    "CNN_ArcFace": lambda: CNNArcFaceBaseline(N_CHANNELS, EMB_DIM),
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_and_split(seed):
    data  = np.load(BED_NPZ, allow_pickle=True)
    X     = data["X"].astype(np.float32)
    y     = data["y"].astype(np.int64)
    runs_raw = data["session"] if "session" in data.files else data["runs"]
    runs  = np.asarray([str(r) for r in runs_raw], dtype=object)

    train_idx = np.where(np.isin(runs, TRAIN_RUNS))[0]
    test_idx  = np.where(np.isin(runs, TEST_RUNS))[0]

    X_tr = X[train_idx]; y_tr = y[train_idx]
    X_te = X[test_idx];  y_te = y[test_idx]

    tr_idx, val_idx = train_test_split(
        np.arange(len(X_tr)), test_size=VAL_SPLIT,
        stratify=y_tr, random_state=seed
    )
    return X_tr, y_tr, tr_idx, val_idx, X_te, y_te


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def extract_embeddings(model, X_np, y_np):
    ds = EEGDataset(X_np, y_np)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    E, Y = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            E.append(model(xb.to(DEVICE)).cpu().numpy())
            Y.append(yb.numpy())
    E = np.concatenate(E); Y = np.concatenate(Y)
    return (E / (np.linalg.norm(E,axis=1,keepdims=True)+1e-12)).astype(np.float32), Y.astype(np.int64)


def evaluate(model, X_enroll, y_enroll, X_test, y_test):
    E_en, Y_en = extract_embeddings(model, X_enroll, y_enroll)
    E_te, Y_te = extract_embeddings(model, X_test,   y_test)

    # KMeans prototypes
    pvecs, powner = [], []
    for sid in sorted(np.unique(Y_en)):
        idx = np.where(Y_en==sid)[0]
        km = KMeans(n_clusters=min(KMEANS_K,len(idx)), random_state=42, n_init=10)
        km.fit(E_en[idx])
        for c in km.cluster_centers_:
            c = c / (np.linalg.norm(c)+1e-12)
            pvecs.append(c.astype(np.float32)); powner.append(int(sid))
    P = np.stack(pvecs); P_owner = np.asarray(powner)

    scores, labels = [], []
    for i in range(len(E_te)):
        e = E_te[i]; tid = Y_te[i]; sim = e @ P.T
        scores.append(float(sim[P_owner==tid].max())); labels.append(1)
        for sid in np.unique(P_owner):
            if sid==tid: continue
            scores.append(float(sim[P_owner==sid].max())); labels.append(0)

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    auc_val = sk_auc(fpr, tpr)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr-fnr))
    eer = (fpr[idx]+fnr[idx])/2.0

    # CRR
    sim_all  = E_te @ P.T
    subj_ids = sorted(np.unique(Y_en))
    sc = np.zeros((len(E_te),len(subj_ids)))
    for j,sid in enumerate(subj_ids):
        sc[:,j] = sim_all[:,P_owner==sid].max(axis=1)
    pred = np.array([subj_ids[j] for j in sc.argmax(axis=1)])
    crr  = float(np.mean(pred==Y_te))*100

    return float(eer*100), float(auc_val), crr


# ─────────────────────────────────────────────────────────────
# TRAIN ONE MODEL / ONE SEED
# ─────────────────────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_and_eval(model_name, model_fn, seed,
                   X_tr, y_tr, tr_idx, val_idx, X_te, y_te):
    set_seed(seed)
    ckpt_dir = OUT_DIR / model_name / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model_fn().to(DEVICE)
    arc   = ArcFaceLayer(EMB_DIM, N_SUBJECTS, ARC_S, ARC_M).to(DEVICE)

    params = sum(p.numel() for p in model.parameters()) + \
             sum(p.numel() for p in arc.parameters())

    train_ds = EEGDataset(X_tr[tr_idx], y_tr[tr_idx])
    val_ds   = EEGDataset(X_tr[val_idx], y_tr[val_idx])
    tr_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                           drop_last=True, num_workers=2, pin_memory=True)
    va_loader = DataLoader(val_ds,   256,         shuffle=False,
                           num_workers=2, pin_memory=True)

    opt    = torch.optim.AdamW(list(model.parameters())+list(arc.parameters()),
                               lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_val=float("inf"); best_ep=0; pat=0; t0=time.time()

    for ep in range(1, EPOCHS+1):
        model.train(); arc.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                z    = model(xb)
                loss = F.cross_entropy(arc(z, yb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval(); arc.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb=xb.to(DEVICE); yb=yb.to(DEVICE)
                z = model(xb)
                va_loss += F.cross_entropy(arc(z,yb),yb).item()
        va_loss /= len(va_loader)

        if va_loss < best_val:
            best_val=va_loss; best_ep=ep; pat=0
            torch.save({"model": model.state_dict(), "arc": arc.state_dict()},
                       ckpt_dir/"best.pt")
        else:
            pat += 1

        flag = "*" if va_loss==best_val else ""
        print(f"  [{model_name} s{seed}] ep {ep:03d} tr={tr_loss:.4f} val={va_loss:.4f} {flag}")
        if pat >= PATIENCE:
            print(f"  Early stop at {ep}"); break

    ckpt = torch.load(ckpt_dir/"best.pt", map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    arc.load_state_dict(ckpt["arc"])

    eer, auc_val, crr = evaluate(model, X_tr, y_tr, X_te, y_te)
    train_min = (time.time()-t0)/60
    print(f"  [{model_name} s{seed}] EER={eer:.4f}% AUC={auc_val:.4f} CRR={crr:.2f}% ({train_min:.1f}min)")

    return {"seed":seed, "best_epoch":best_ep, "train_min":train_min,
            "params":params, "eer_percent":eer, "auc":auc_val, "crr_percent":crr}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("  BED BASELINES — EEGNet / DeepConvNet / CNN+ArcFace")
    print("="*60)

    all_summaries = {}

    for model_name, model_fn in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"  MODEL: {model_name}")
        seed_results = []

        for seed in SEEDS:
            X_tr, y_tr, tr_idx, val_idx, X_te, y_te = load_and_split(seed)
            res = train_and_eval(model_name, model_fn, seed,
                                 X_tr, y_tr, tr_idx, val_idx, X_te, y_te)
            seed_results.append(res)

        eers = [r["eer_percent"] for r in seed_results]
        aucs = [r["auc"]         for r in seed_results]
        crrs = [r["crr_percent"] for r in seed_results]

        summary = {
            "model": model_name,
            "per_seed": seed_results,
            "eer_mean": float(np.mean(eers)),
            "eer_std":  float(np.std(eers, ddof=1)),
            "auc_mean": float(np.mean(aucs)),
            "crr_mean": float(np.mean(crrs)),
        }
        all_summaries[model_name] = summary

        with open(OUT_DIR / f"{model_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n  {model_name} 5-seed: EER={np.mean(eers):.4f}%±{np.std(eers,ddof=1):.4f}%"
              f"  AUC={np.mean(aucs):.4f}  CRR={np.mean(crrs):.2f}%")

    # ── Combined table ──
    print("\n" + "="*60)
    print("  BED COMPARISON TABLE (for Table S3 supplementary)")
    print("="*60)
    print(f"  {'Model':<18} {'EER (%)':>12}  {'AUC':>8}  {'CRR (%)':>10}")
    print(f"  {'-'*52}")

    # Load DOMCS result (produced by bed_domcs_locked.py)
    domcs_json = OUT_DIR.parent / "BED_DOMCS_LOCKED_RESULTS" / "BED_DOMCS_LOCKED_SUMMARY.json"
    if domcs_json.exists():
        with open(domcs_json) as f:
            d = json.load(f)
        print(f"  {'DOMCS-EEG (ours)':<18} "
              f"{d['eer_mean']:>8.4f}±{d['eer_std']:.4f}  "
              f"{d['auc_mean']:>8.4f}  {d['crr_mean']:>10.2f}")

    for name, s in all_summaries.items():
        print(f"  {name:<18} "
              f"{s['eer_mean']:>8.4f}±{s['eer_std']:.4f}  "
              f"{s['auc_mean']:>8.4f}  {s['crr_mean']:>10.2f}")

    with open(OUT_DIR / "BED_COMPARISON_SUMMARY.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n  Saved: {OUT_DIR}/BED_COMPARISON_SUMMARY.json")
    print("="*60)


if __name__ == "__main__":
    main()
