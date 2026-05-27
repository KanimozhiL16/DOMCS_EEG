#!/usr/bin/env python3
"""
09_ablation_study.py — DOMCS-EEG Ablation Study
=================================================
Runs 4 ablation configurations from scratch using seed=3 (best seed).
Each config removes one key component and measures EER degradation.

Fills TABLE_02 with real EER / AUC numbers.

Ablation configurations:
  A1: Full DOMCS-EEG (reference — matches locked version seed=3)
  A2: w/o Orthogonality loss (λ_orth = 0)
  A3: w/o Supervised Contrastive loss (λ_sc = 0)
  A4: w/o State branch (no state supervision, no orth)
  A5: w/o Dual-space (identity loss on ALL windows, not REST-only)
      → This reproduces the v1 leakage bug — expect near-0% EER

CRITICAL: Run from Brev with GPU available.
  cd /home/nvidia/24PHD1237/DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/code
  python 09_ablation_study.py

Output:
  logs/ablation_results.json
  tables/TABLE_02_ablation.csv
  latex_tables/TABLE_02_ablation.tex
"""

import os, sys, json, csv, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (N_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, LR_MIN,
                    N_SUBJECTS, N_STATE_CLASSES, LOG_DIR, CKPT_DIR,
                    TABLE_DIR, LATEX_DIR, NPZ_PATH,
                    LAMBDA_SUPCON, LAMBDA_STATE, LAMBDA_ORTH)
from model       import DOMCSEEGModel, ArcFaceHead, StateClassifier
from losses      import ArcFaceLoss, SupConLoss, StateLoss, OrthogonalityLoss
from data_loader import (load_npz, prepare_training_data, create_dataloaders,
                          build_gallery, get_verification_set, score_probe, compute_eer)

ABLATION_SEED    = 3
ABLATION_EPOCHS  = 60    # full training — same as locked version
ABLATION_DIR     = Path(LOG_DIR) / "ablation"


# ─── Ablation Configs ──────────────────────────────────────────────────────────

CONFIGS = {
    "A1_full_domcs": {
        "description":   "Full DOMCS-EEG (reference)",
        "lambda_sc":     LAMBDA_SUPCON,
        "lambda_state":  LAMBDA_STATE,
        "lambda_orth":   LAMBDA_ORTH,
        "identity_on_rest_only": True,
        "use_state_branch": True,
    },
    "A2_no_orth": {
        "description":   "w/o Orthogonality loss",
        "lambda_sc":     LAMBDA_SUPCON,
        "lambda_state":  LAMBDA_STATE,
        "lambda_orth":   0.0,
        "identity_on_rest_only": True,
        "use_state_branch": True,
    },
    "A3_no_supcon": {
        "description":   "w/o Supervised Contrastive loss",
        "lambda_sc":     0.0,
        "lambda_state":  LAMBDA_STATE,
        "lambda_orth":   LAMBDA_ORTH,
        "identity_on_rest_only": True,
        "use_state_branch": True,
    },
    "A4_no_state": {
        "description":   "w/o State branch (no state/orth losses)",
        "lambda_sc":     LAMBDA_SUPCON,
        "lambda_state":  0.0,
        "lambda_orth":   0.0,
        "identity_on_rest_only": True,
        "use_state_branch": False,
    },
    "A5_no_dual_space": {
        "description":   "w/o B2T-safe split (identity loss on ALL windows)",
        "lambda_sc":     LAMBDA_SUPCON,
        "lambda_state":  LAMBDA_STATE,
        "lambda_orth":   LAMBDA_ORTH,
        "identity_on_rest_only": False,   # LEAKAGE — reproduces v1 bug
        "use_state_branch": True,
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device('cpu')
        print("  Running on CPU — ablation will be slow")
    return dev


# ─── Single ablation training run ─────────────────────────────────────────────

def train_ablation_config(cfg_name, cfg, X, y, session, device):
    """
    Train one ablation config and return EER and AUC.
    Returns (eer, roc_auc) as floats.
    """
    set_seed(ABLATION_SEED)
    print(f"\n  ─── {cfg_name}: {cfg['description']} ───")

    # Datasets and loaders
    train_ds, val_ds, stats = prepare_training_data(X, y, session, seed=ABLATION_SEED)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=BATCH_SIZE)

    # Model
    model     = DOMCSEEGModel().to(device)
    arc_head  = ArcFaceHead(n_classes=N_SUBJECTS).to(device)
    state_cls = StateClassifier(n_classes=N_STATE_CLASSES).to(device)

    # Losses
    arc_loss_fn   = ArcFaceLoss()
    sc_loss_fn    = SupConLoss(temperature=0.07)
    state_loss_fn = StateLoss()
    orth_loss_fn  = OrthogonalityLoss()

    # Optimizer (all active params)
    params = list(model.parameters()) + list(arc_head.parameters())
    if cfg["use_state_branch"]:
        params += list(state_cls.parameters())
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=ABLATION_EPOCHS, eta_min=LR_MIN)

    # Training loop
    t0 = time.time()
    for epoch in range(1, ABLATION_EPOCHS + 1):
        model.train(); arc_head.train(); state_cls.train()
        total_loss = 0.0
        n_batches  = 0

        for x, y_subj, y_state in train_loader:
            x      = x.to(device)
            y_subj = y_subj.to(device)
            y_state= y_state.to(device)

            z_id, z_state, _ = model(x)
            optimizer.zero_grad()

            if cfg["identity_on_rest_only"]:
                rest_mask = (y_state == 0)
            else:
                rest_mask = torch.ones(len(y_state), dtype=torch.bool, device=device)

            # Identity losses: REST only (or ALL for A5)
            if rest_mask.sum() > 1:
                arc_logits_r = arc_head(z_id[rest_mask], y_subj[rest_mask])
                L_arc = arc_loss_fn(arc_logits_r, y_subj[rest_mask])
                L_sc  = sc_loss_fn(z_id[rest_mask], y_subj[rest_mask]) * cfg["lambda_sc"]
            else:
                L_arc = torch.tensor(0.0, device=device, requires_grad=True)
                L_sc  = torch.tensor(0.0, device=device)

            # State + orth losses: ALL windows (if enabled)
            if cfg["use_state_branch"] and cfg["lambda_state"] > 0:
                state_logits = state_cls(z_state)
                L_state = state_loss_fn(state_logits, y_state) * cfg["lambda_state"]
            else:
                L_state = torch.tensor(0.0, device=device)

            if cfg["use_state_branch"] and cfg["lambda_orth"] > 0:
                L_orth = orth_loss_fn(z_id, z_state) * cfg["lambda_orth"]
            else:
                L_orth = torch.tensor(0.0, device=device)

            loss = L_arc + L_sc + L_state + L_orth
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == ABLATION_EPOCHS:
            avg = total_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:3d}/{ABLATION_EPOCHS}  loss={avg:.4f}  "
                  f"elapsed={elapsed/60:.1f}min")

    # Evaluate: build gallery and compute EER
    print(f"  Evaluating {cfg_name}...")
    gallery = build_gallery(model, X, y, session, device)
    X_test, y_test, sess_test = get_verification_set(X, y, session)

    # Extract embeddings for test set
    model.eval()
    all_emb = []
    with torch.no_grad():
        bs = 512
        for i in range(0, len(X_test), bs):
            xb = torch.from_numpy(X_test[i:i+bs]).to(device)
            z  = model.get_identity_embedding(xb)
            all_emb.append(z.cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)

    # Collect genuine and impostor scores
    genuine, impostor = [], []
    subjects = np.unique(y_test)
    for subj in subjects:
        mask = y_test == subj
        probes = all_emb[mask]
        for p in probes:
            scores = score_probe(p, gallery)
            genuine.append(scores.get(int(subj), 0.0))
            for k, v in scores.items():
                if k != int(subj):
                    impostor.append(v)

    eer, roc_auc, _, _, _ = compute_eer(np.array(genuine), np.array(impostor))
    print(f"  ✓ {cfg_name}: EER={eer:.4f}%  AUC={roc_auc:.4f}")
    return float(eer), float(roc_auc)


# ─── Table generation ─────────────────────────────────────────────────────────

def save_results(results):
    """Save ablation results to JSON, CSV and LaTeX."""
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = ABLATION_DIR / "ablation_results.json"
    with open(str(json_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ JSON: {json_path}")

    # CSV → TABLE_02
    csv_rows = []
    ref_eer = None
    for r in results:
        if "A1" in r["config"]:
            ref_eer = r["eer"]
        delta = f"+{r['eer'] - ref_eer:.2f}" if ref_eer and r["eer"] != ref_eer else "—"
        csv_rows.append({
            "Config":        r["config"],
            "Description":   r["description"],
            "EER (%)":       f"{r['eer']:.4f}",
            "AUC":           f"{r['auc']:.4f}",
            "ΔEER vs Full":  delta,
        })

    csv_path = Path(TABLE_DIR) / "TABLE_02_ablation.csv"
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    with open(str(csv_path), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Config","Description","EER (%)","AUC","ΔEER vs Full"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  ✓ CSV: {csv_path}")

    # LaTeX → TABLE_02
    Path(LATEX_DIR).mkdir(parents=True, exist_ok=True)
    tex_path = Path(LATEX_DIR) / "TABLE_02_ablation.tex"
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Ablation study — contribution of each DOMCS-EEG loss component.",
        r"A1 = full model (reference); A2--A5 remove one component each.",
        r"EER (\%) reported for B2T evaluation protocol, seed=3.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"\textbf{Config} & \textbf{Description} & \textbf{EER (\%)} & \textbf{AUC} & \textbf{$\Delta$EER} \\",
        r"\hline",
    ]
    ref_eer = None
    for r in results:
        if "A1" in r["config"]:
            ref_eer = r["eer"]
        delta_str = f"+{r['eer'] - ref_eer:.2f}" if ref_eer and r["eer"] != ref_eer else "—"
        lines.append(
            f"{r['config']} & {r['description']} & {r['eer']:.4f} & {r['auc']:.4f} & {delta_str} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(str(tex_path), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ LaTeX: {tex_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DOMCS-EEG Ablation Study")
    print(f"Seed: {ABLATION_SEED}  |  Epochs: {ABLATION_EPOCHS}")
    print("=" * 60)

    device = get_device()

    print("\nLoading data...")
    X, y, session = load_npz(NPZ_PATH)

    results = []
    for cfg_name, cfg in CONFIGS.items():
        eer, roc_auc = train_ablation_config(cfg_name, cfg, X, y, session, device)
        results.append({
            "config":      cfg_name,
            "description": cfg["description"],
            "seed":        ABLATION_SEED,
            "epochs":      ABLATION_EPOCHS,
            "eer":         eer,
            "auc":         roc_auc,
        })

    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    ref = next(r for r in results if "A1" in r["config"])
    for r in results:
        delta = f"  Δ={r['eer'] - ref['eer']:+.4f}%" if r["config"] != ref["config"] else ""
        print(f"  {r['config']}: EER={r['eer']:.4f}%  AUC={r['auc']:.4f}{delta}")

    save_results(results)
    print("\n✓ Ablation study complete. Copy TABLE_02_ablation.csv/.tex to paper.")


if __name__ == "__main__":
    main()
