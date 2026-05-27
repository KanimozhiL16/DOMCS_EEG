#!/usr/bin/env python3
"""
01_train_dual_space_domcs.py — CORRECTED v2
============================================
SCIENTIFIC FIX: B2T-safe dual-space training.

KEY RULE (prevents data leakage):
  ArcFace + SupCon identity losses → REST WINDOWS ONLY (R01+R02)
  State classification loss         → REST + TASK WINDOWS (R01-R14)
  Orthogonality loss                → ALL WINDOWS

This ensures:
  - Identity encoder never sees R03-R14 with subject labels during training
  - R03-R14 used ONLY for state supervision (no identity label leakage)
  - B2T evaluation integrity preserved: EER will be realistic

WHY EER=0% HAPPENED (v1 bug):
  v1 passed ArcFace+SupCon losses on ALL windows including R03-R14.
  Model memorised R03-R14 subject identity → evaluated on same R03-R14 → EER=0%.

TRAINING DATA FLOW:
  Batch may contain REST and TASK windows (mixed).
  - REST windows (y_state==0): ArcFace + SupCon + State + Orth losses
  - TASK windows (y_state==1): State + Orth losses ONLY
"""

import os, sys, json, time, csv
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (SEEDS, N_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, LR_MIN,
                    N_SUBJECTS, N_STATE_CLASSES, CKPT_DIR, LOG_DIR, VERIF_DIR,
                    LAMBDA_SUPCON, LAMBDA_STATE, LAMBDA_ORTH)
from model       import DOMCSEEGModel, ArcFaceHead, StateClassifier, count_parameters
from losses      import ArcFaceLoss, SupConLoss, StateLoss, OrthogonalityLoss
from data_loader import (load_npz, prepare_training_data, create_dataloaders,
                          write_verification_log)


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
        print("  Running on CPU")
    return dev


def train_one_epoch(model, arc_head, state_cls,
                    arc_loss_fn, sc_loss_fn, state_loss_fn, orth_loss_fn,
                    loader, optimizer, device):
    """
    B2T-safe training step.
    - REST windows (y_state==0): ALL losses
    - TASK windows (y_state==1): state + orth losses ONLY
    """
    model.train(); arc_head.train(); state_cls.train()

    totals = {"L_total": 0, "L_arcface": 0, "L_supcon": 0,
              "L_state": 0, "L_orth": 0}
    n_batches = 0

    for x, y_subj, y_state in loader:
        x      = x.to(device)
        y_subj = y_subj.to(device)
        y_state= y_state.to(device)

        optimizer.zero_grad(set_to_none=True)

        z_id, z_state_emb, f = model(x)
        state_logits = state_cls(z_state_emb)

        # ── Masks ──────────────────────────────────────────────
        rest_mask = (y_state == 0)   # R01+R02 only
        task_mask = (y_state == 1)   # R03-R14 only

        loss = torch.tensor(0.0, device=device)

        # ── Identity losses: REST ONLY ──────────────────────────
        if rest_mask.sum() > 1:
            z_id_rest   = z_id[rest_mask]
            y_subj_rest = y_subj[rest_mask]
            arc_logits  = arc_head(z_id_rest, y_subj_rest)

            L_arc = arc_loss_fn(arc_logits, y_subj_rest)
            L_sc  = sc_loss_fn(z_id_rest, y_subj_rest)
            loss  = loss + L_arc + LAMBDA_SUPCON * L_sc

            totals["L_arcface"] += L_arc.item()
            totals["L_supcon"]  += L_sc.item()

        # ── State loss: ALL windows ─────────────────────────────
        L_state = state_loss_fn(state_logits, y_state)
        loss    = loss + LAMBDA_STATE * L_state
        totals["L_state"] += L_state.item()

        # ── Orthogonality: ALL windows ──────────────────────────
        L_orth = orth_loss_fn(z_id, z_state_emb)
        loss   = loss + LAMBDA_ORTH * L_orth
        totals["L_orth"] += L_orth.item()

        totals["L_total"] += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) +
            list(arc_head.parameters()) +
            list(state_cls.parameters()), max_norm=1.0)
        optimizer.step()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, arc_head, state_cls,
             arc_loss_fn, sc_loss_fn, state_loss_fn, orth_loss_fn,
             loader, device):
    model.eval(); arc_head.eval(); state_cls.eval()

    totals = {"L_total": 0, "L_arcface": 0, "L_supcon": 0,
              "L_state": 0, "L_orth": 0}
    n_batches   = 0
    state_corr  = 0; state_tot = 0

    for x, y_subj, y_state in loader:
        x      = x.to(device)
        y_subj = y_subj.to(device)
        y_state= y_state.to(device)

        z_id, z_state_emb, f = model(x)
        state_logits = state_cls(z_state_emb)
        rest_mask    = (y_state == 0)

        loss = torch.tensor(0.0, device=device)
        if rest_mask.sum() > 1:
            z_id_r = z_id[rest_mask]; y_r = y_subj[rest_mask]
            arc_logits = arc_head(z_id_r, y_r)
            L_arc = arc_loss_fn(arc_logits, y_r)
            L_sc  = sc_loss_fn(z_id_r, y_r)
            loss  = loss + L_arc + LAMBDA_SUPCON * L_sc
            totals["L_arcface"] += L_arc.item()
            totals["L_supcon"]  += L_sc.item()

        L_state = state_loss_fn(state_logits, y_state)
        L_orth  = orth_loss_fn(z_id, z_state_emb)
        loss    = loss + LAMBDA_STATE * L_state + LAMBDA_ORTH * L_orth
        totals["L_state"] += L_state.item()
        totals["L_orth"]  += L_orth.item()
        totals["L_total"] += loss.item()

        preds = state_logits.argmax(dim=1)
        state_corr += (preds == y_state).sum().item()
        state_tot  += len(y_state)
        n_batches  += 1

    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    metrics["state_acc"] = state_corr / state_tot if state_tot > 0 else 0.0
    return metrics


@torch.no_grad()
def save_embeddings(model, loader, device, out_path):
    model.eval()
    all_zid, all_zst, all_ys, all_ystate = [], [], [], []
    for x, y_subj, y_state in loader:
        z_id, z_state, _ = model(x.to(device))
        all_zid.append(z_id.cpu().numpy())
        all_zst.append(z_state.cpu().numpy())
        all_ys.append(y_subj.numpy())
        all_ystate.append(y_state.numpy())
    np.savez(str(out_path),
             z_id    = np.concatenate(all_zid),
             z_state = np.concatenate(all_zst),
             y_subj  = np.concatenate(all_ys),
             y_state = np.concatenate(all_ystate))


def train_seed(seed, X, y, session, device):
    print(f"\n{'='*60}")
    print(f"  TRAINING v2 (B2T-safe) — Seed {seed}")
    print(f"{'='*60}")
    set_seed(seed)

    ckpt_dir = Path(CKPT_DIR) / f"seed_{seed}"
    log_dir  = Path(LOG_DIR)  / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)

    train_ds, val_ds, stats = prepare_training_data(X, y, session,
                                                    val_split=0.10, seed=seed)
    write_verification_log(stats, seed=seed)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds,
                                                  batch_size=BATCH_SIZE)

    # Count REST vs TASK in training set
    y_state_arr = train_ds.y_state.numpy()
    n_rest_tr = int((y_state_arr == 0).sum())
    n_task_tr = int((y_state_arr == 1).sum())

    print(f"  Train: {len(train_ds):,} total  |  REST: {n_rest_tr:,}  TASK: {n_task_tr:,}")
    print(f"  Val:   {len(val_ds):,} windows")
    print(f"  IDENTITY losses applied to: REST windows ONLY ({n_rest_tr:,})")
    print(f"  STATE   losses applied to: ALL windows ({len(train_ds):,})")

    model     = DOMCSEEGModel().to(device)
    arc_head  = ArcFaceHead(n_classes=N_SUBJECTS).to(device)
    state_cls = StateClassifier(n_classes=N_STATE_CLASSES).to(device)

    total_params, _ = count_parameters(model)
    print(f"  Model parameters: {total_params:,}")

    params = (list(model.parameters()) +
              list(arc_head.parameters()) +
              list(state_cls.parameters()))
    optimizer = optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR_MIN)

    arc_loss_fn   = ArcFaceLoss()
    sc_loss_fn    = SupConLoss(temperature=0.07)
    state_loss_fn = StateLoss()
    orth_loss_fn  = OrthogonalityLoss()

    csv_path   = log_dir / "train_log.csv"
    fieldnames = ["epoch","lr",
                  "train_L_total","train_L_arcface","train_L_supcon",
                  "train_L_state","train_L_orth",
                  "val_L_total","val_state_acc"]

    best_val_loss = float('inf')
    best_epoch    = -1

    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, N_EPOCHS + 1):
            t0 = time.time()

            train_m = train_one_epoch(
                model, arc_head, state_cls,
                arc_loss_fn, sc_loss_fn, state_loss_fn, orth_loss_fn,
                train_loader, optimizer, device)
            val_m = validate(
                model, arc_head, state_cls,
                arc_loss_fn, sc_loss_fn, state_loss_fn, orth_loss_fn,
                val_loader, device)
            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]

            row = {"epoch": epoch, "lr": f"{lr_now:.2e}",
                   **{f"train_{k}": f"{v:.5f}" for k, v in train_m.items()},
                   **{f"val_{k}":   f"{v:.5f}" for k, v in
                      {k: val_m[k] for k in ["L_total","state_acc"]}.items()}}
            writer.writerow(row); csvf.flush()

            elapsed = time.time() - t0
            print(f"  Ep {epoch:3d}/{N_EPOCHS} | "
                  f"tr={train_m['L_total']:.4f} "
                  f"vl={val_m['L_total']:.4f} "
                  f"st_acc={val_m['state_acc']:.3f} "
                  f"lr={lr_now:.2e} [{elapsed:.1f}s]")

            if val_m["L_total"] < best_val_loss:
                best_val_loss = val_m["L_total"]
                best_epoch    = epoch
                torch.save({
                    "epoch": epoch, "seed": seed,
                    "val_loss": best_val_loss,
                    "model_state":     model.state_dict(),
                    "arc_state":       arc_head.state_dict(),
                    "state_cls_state": state_cls.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "stats":           stats,
                    "training_version": "v2_b2t_safe",
                    "identity_loss_data": "REST_ONLY",
                    "state_loss_data":    "REST+TASK",
                }, str(ckpt_dir / "model_best.pt"))

    print(f"\n  ✓ Best checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.5f}")

    # Save best-seed embeddings
    ckpt = torch.load(str(ckpt_dir / "model_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    emb_path = log_dir / "embeddings_best.npz"
    save_embeddings(model, val_loader, device, emb_path)
    print(f"  ✓ Embeddings saved: {emb_path}")

    summary = {
        "seed": seed, "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "n_rest_train": n_rest_tr, "n_task_train": n_task_tr,
        "identity_loss_applied_to": "REST_ONLY (R01+R02)",
        "state_loss_applied_to":    "ALL (R01-R14)",
        "b2t_safe": True,
    }
    with open(log_dir / "seed_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    print("DOMCS-EEG — Dual-Space Training v2 (B2T-safe)")
    print("="*60)
    print("IDENTITY losses: REST windows only (R01+R02)")
    print("STATE    losses: ALL windows (R01-R14)")
    print("="*60)
    device = get_device()

    from config import NPZ_PATH
    X, y, session = load_npz(NPZ_PATH)

    all_summaries = []
    for seed in SEEDS:
        summary = train_seed(seed, X, y, session, device)
        all_summaries.append(summary)

    global_out = Path(LOG_DIR) / "training_summary_all_seeds.json"
    with open(global_out, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE — All seeds (B2T-safe v2)")
    for s in all_summaries:
        print(f"  Seed {s['seed']}: best_epoch={s['best_epoch']}  "
              f"val_loss={s['best_val_loss']:.5f}")
    print("="*60)


if __name__ == "__main__":
    main()
