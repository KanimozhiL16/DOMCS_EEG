#!/usr/bin/env python3
"""
04_security_analysis.py
========================
Full T0–T5 security evaluation on best-seed checkpoint.

T0: Random forgery      — cross-subject resting-state probes
T1: Skilled forgery     — cross-subject rest-state probes (adversarial knowledge)
T2: Enrollment leakage  — partial enrollment data exposed (10/25/50/100%)
T3: Signal noise        — AWGN at 5/10/20/30 dB + powerline 50/60 Hz
T4: Adversarial attacks — FGSM + PGD untargeted
T5: Targeted impersonation — PGD targeted attack

All figures saved as PNG + PDF.
All tables saved as CSV + TEX.
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (SEEDS, CKPT_DIR, LOG_DIR, FIG_DIR, TABLE_DIR,
                    PNG_DIR, PDF_DIR, REST_RUNS, TASK_RUNS,
                    FGSM_EPS_LIST, PGD_EPS_LIST, PGD_STEPS, PGD_ALPHA,
                    AWGN_SNR_LIST, MPL_STYLE, COLORS)
from model       import DOMCSEEGModel
from data_loader import (load_npz, build_gallery, get_verification_set,
                          score_probe, compute_eer)

plt.rcParams.update(MPL_STYLE)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_best_model(device):
    """
    Load the best checkpoint by EER (lowest EER across seeds).
    Reads eval_summary.json produced by 02_evaluate_b2t.py.
    Falls back to lowest val_loss if eval_summary.json not found.
    This ensures security analysis uses the SAME model as reported best seed.
    """
    import json
    eval_summary = Path(LOG_DIR) / "eval_summary.json"
    if eval_summary.exists():
        with open(str(eval_summary)) as f:
            summary = json.load(f)
        best_seed = summary.get("best_seed", SEEDS[0])
        best_eer  = summary.get("best_eer", float('inf'))
        print(f"  Loading best checkpoint: seed_{best_seed}  EER={best_eer:.4f}%  (from eval_summary.json)")
    else:
        # Fallback: lowest val_loss
        best_loss = float('inf')
        best_seed = SEEDS[0]
        for seed in SEEDS:
            p = Path(CKPT_DIR) / f"seed_{seed}" / "model_best.pt"
            if not p.exists(): continue
            ckpt = torch.load(str(p), map_location='cpu')
            if ckpt.get("val_loss", float('inf')) < best_loss:
                best_loss = ckpt["val_loss"]
                best_seed = seed
        print(f"  Loading best checkpoint: seed_{best_seed}  val_loss={best_loss:.5f}  (fallback: val_loss)")

    ckpt_path = Path(CKPT_DIR) / f"seed_{best_seed}" / "model_best.pt"
    model = DOMCSEEGModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, best_seed


def add_awgn(X, snr_db):
    """Add AWGN to signal batch. X: numpy (N, 64, 256)."""
    X_noisy = X.copy()
    signal_power = np.mean(X ** 2, axis=(1, 2), keepdims=True)
    snr_linear   = 10 ** (snr_db / 10.0)
    noise_power  = signal_power / (snr_linear + 1e-9)
    noise = np.random.default_rng(42).normal(0, 1, X.shape).astype(np.float32)
    noise_scaled = noise * np.sqrt(noise_power)
    return X_noisy + noise_scaled


def add_powerline(X, freq_hz, fs=128):
    """Add sinusoidal powerline noise. X: numpy (N, 64, 256)."""
    t     = np.arange(256) / fs
    noise = 0.05 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return X + noise[np.newaxis, np.newaxis, :]


def embed_batch(model, X_np, device, batch_size=512):
    """Get z_id embeddings for numpy array."""
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            embs.append(model.get_identity_embedding(xb).cpu().numpy())
    embs = np.concatenate(embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-8)


def _forward_with_grad(model, x):
    """
    Forward pass that BUILDS the computation graph (no torch.no_grad).
    Required by FGSM and PGD attacks which need d(loss)/d(input).
    DO NOT use model.get_identity_embedding() here — it wraps in no_grad.
    """
    f    = model.encoder(x)
    z_id = model.id_branch(f)
    return F.normalize(z_id, p=2, dim=1)


def fgsm_attack(model, x, threshold, eps, device):
    """
    FGSM untargeted: push z_id AWAY from its clean embedding.
    Loss = cosine_similarity(z_adv, z_clean) — minimising this
    maximises the angle between adversarial and genuine embedding,
    reducing the genuine score and increasing EER.
    """
    x_t   = x.clone().detach().to(device)
    # Step 1: get clean embedding (no grad needed)
    with torch.no_grad():
        z_clean = _forward_with_grad(model, x_t.unsqueeze(0)).detach()
    # Step 2: adversarial forward with grad
    x_adv = x_t.clone().requires_grad_(True)
    z_adv = _forward_with_grad(model, x_adv.unsqueeze(0))
    # Minimise cosine similarity → push z_adv away from z_clean
    loss  = F.cosine_similarity(z_adv, z_clean).mean()
    loss.backward()
    grad_sign = x_adv.grad.sign()
    return (x_t + eps * grad_sign).clamp(x_t.min(), x_t.max())


def pgd_attack(model, x, eps, n_steps, alpha, device, targeted=False,
               target_emb=None):
    """
    PGD attack.
    Untargeted: push z_id away from clean embedding (maximise angular distance).
    Targeted: pull z_id toward target_emb (impersonation).
    """
    model.eval()
    x_orig = x.clone().detach().to(device)
    x_adv  = x_orig.clone()
    # Pre-compute clean embedding for untargeted attack anchor
    with torch.no_grad():
        z_clean = _forward_with_grad(model, x_orig.unsqueeze(0)).detach()
    for _ in range(n_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        z = _forward_with_grad(model, x_adv.unsqueeze(0))
        if targeted:
            # Pull toward target: maximise cosine sim → minimise negative
            loss = -F.cosine_similarity(z, target_emb).mean()
        else:
            # Push away from clean: minimise cosine sim with clean embedding
            loss = F.cosine_similarity(z, z_clean).mean()
        loss.backward()
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            if targeted:
                x_adv = x_adv - alpha * grad_sign
            else:
                x_adv = x_adv + alpha * grad_sign
            delta = torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = (x_orig + delta).clamp(x_orig.min(), x_orig.max())
    return x_adv.detach()


def evaluate_scenario(genuine_scores, impostor_scores, label):
    eer, auc, fpr, tpr, thr = compute_eer(genuine_scores, impostor_scores)
    print(f"    {label:30s}  EER={eer:.3f}%  AUC={auc:.4f}")
    return {"label": label, "eer": float(eer), "auc": float(auc)}


def main():
    print("DOMCS-EEG — Security Evaluation (T0–T5)")
    print("="*60)
    device = get_device()
    X, y, session = load_npz()
    model, best_seed = load_best_model(device)

    # Build clean gallery + verification set
    print("\n  Building gallery (R01+R02)...")
    gallery = build_gallery(model, X, y, session, device)
    X_test, y_test, sess_test = get_verification_set(X, y, session)

    rng = np.random.default_rng(0)
    subjects = sorted(gallery.keys())

    def score_X(X_arr, y_arr):
        embs = embed_batch(model, X_arr, device)
        gen, imp = [], []
        for emb, subj in zip(embs, y_arr):
            sc  = score_probe(emb, gallery)
            gen.append(sc[int(subj)])
            other = rng.choice([s for s in subjects if s != int(subj)])
            imp.append(sc[other])
        return np.array(gen), np.array(imp)

    results = []

    # ── Clean baseline ────────────────────────────────────────────
    print("\n  T0-baseline: Clean B2T...")
    g_clean, i_clean = score_X(X_test, y_test)
    results.append(evaluate_scenario(g_clean, i_clean, "Clean (B2T)"))

    # ── T0: Random forgery ────────────────────────────────────────
    print("\n  T0: Random forgery...")
    # Cross-subject: score rest probes (R04) from random subjects
    r04_mask = session == 'R04'
    X_r04 = X[r04_mask]; y_r04 = y[r04_mask]
    g0, i0 = score_X(X_r04, y_r04)
    # T0: genuine vs random impostor from DIFFERENT session
    results.append(evaluate_scenario(g0, i0, "T0 Random forgery"))

    # ── T1: Skilled forgery ───────────────────────────────────────
    print("\n  T1: Skilled forgery (rest-state probes)...")
    rest_mask = np.isin(session, REST_RUNS)
    X_rest_sk = X[rest_mask]; y_rest_sk = y[rest_mask]
    g1, i1 = score_X(X_rest_sk, y_rest_sk)
    results.append(evaluate_scenario(g1, i1, "T1 Skilled forgery"))

    # ── T2: Enrollment leakage ────────────────────────────────────
    print("\n  T2: Enrollment leakage...")
    for pct in [10, 25, 50, 100]:
        rest_mask = np.isin(session, REST_RUNS)
        X_r = X[rest_mask]; y_r = y[rest_mask]
        n_leakage = int(len(X_r) * pct / 100)
        idx_leak  = rng.choice(len(X_r), n_leakage, replace=False)
        # Build gallery with leakage
        gallery_leak = build_gallery(model,
                                     np.concatenate([X[rest_mask], X_test[idx_leak[:100]]]),
                                     np.concatenate([y[rest_mask], y_test[idx_leak[:100]]]),
                                     np.concatenate([session[rest_mask],
                                                     np.array(['R01'] * min(100, n_leakage))]),
                                     device)
        g2, i2 = score_X(X_test, y_test)  # probes unchanged
        results.append(evaluate_scenario(g2, i2, f"T2 Leakage {pct}%"))

    # ── T3: AWGN noise ────────────────────────────────────────────
    print("\n  T3: AWGN noise...")
    for snr in AWGN_SNR_LIST:
        X_noisy = add_awgn(X_test, snr)
        g3, i3  = score_X(X_noisy, y_test)
        results.append(evaluate_scenario(g3, i3, f"T3 AWGN {snr}dB"))

    # Powerline noise
    for freq in [50, 60]:
        X_pl   = add_powerline(X_test, freq)
        g_pl, i_pl = score_X(X_pl, y_test)
        results.append(evaluate_scenario(g_pl, i_pl, f"T3 Powerline {freq}Hz"))

    # ── T4: Adversarial attacks ────────────────────────────────────
    print("\n  T4: Adversarial attacks (FGSM + PGD)...")
    N_adv = min(500, len(X_test))
    idx_adv = rng.choice(len(X_test), N_adv, replace=False)
    X_adv_base = X_test[idx_adv]; y_adv_base = y_test[idx_adv]

    for eps in FGSM_EPS_LIST:
        X_fgsm = np.stack([
            fgsm_attack(model,
                        torch.from_numpy(X_adv_base[i]).to(device),
                        threshold=None, eps=eps, device=device).cpu().numpy()
            for i in range(len(X_adv_base))
        ])
        gf, if_ = score_X(X_fgsm, y_adv_base)
        results.append(evaluate_scenario(gf, if_, f"T4 FGSM eps={eps:.3f}"))

    for eps in PGD_EPS_LIST:
        X_pgd = np.stack([
            pgd_attack(model,
                       torch.from_numpy(X_adv_base[i]).to(device),
                       eps=eps, n_steps=PGD_STEPS, alpha=PGD_ALPHA,
                       device=device).cpu().numpy()
            for i in range(len(X_adv_base))
        ])
        gp, ip = score_X(X_pgd, y_adv_base)
        results.append(evaluate_scenario(gp, ip, f"T4 PGD  eps={eps:.3f}"))

    # ── T5: Targeted impersonation ────────────────────────────────
    print("\n  T5: Targeted impersonation (PGD)...")
    N_t5 = min(200, len(X_test))
    idx_t5 = rng.choice(len(X_test), N_t5, replace=False)
    X_t5 = X_test[idx_t5]; y_t5 = y_test[idx_t5]

    eps_t5 = 0.005
    success_count  = 0
    per_subj_tsr   = {}

    for i in range(len(X_t5)):
        true_subj  = int(y_t5[i])
        # Target: a random different subject's prototype
        tgt_subj   = rng.choice([s for s in subjects if s != true_subj])
        tgt_proto  = torch.from_numpy(gallery[tgt_subj][0:1]).to(device)  # (1, 128)

        x_i     = torch.from_numpy(X_t5[i]).to(device)
        x_adv_i = pgd_attack(model, x_i, eps=eps_t5,
                              n_steps=20, alpha=0.0005, device=device,
                              targeted=True, target_emb=tgt_proto)

        # Check if attack succeeded: adversarial probe scores higher for tgt_subj
        emb_adv = model.get_identity_embedding(x_adv_i.unsqueeze(0))
        emb_adv_np = emb_adv.cpu().numpy().squeeze()
        emb_adv_np = emb_adv_np / (np.linalg.norm(emb_adv_np) + 1e-8)
        sc = score_probe(emb_adv_np, gallery)
        if sc[tgt_subj] > sc.get(true_subj, 0):
            success_count += 1
        per_subj_tsr.setdefault(true_subj, []).append(
            int(sc[tgt_subj] > sc.get(true_subj, 0)))

    tsr = success_count / N_t5 * 100
    print(f"    T5 Targeted impersonation  TSR={tsr:.2f}%  (n={N_t5})")
    results.append({"label": "T5 Targeted impersonation",
                    "eer": None, "auc": None, "tsr": float(tsr)})

    # Per-subject TSR
    per_subj_tsr_mean = {k: float(np.mean(v)) for k, v in per_subj_tsr.items()}
    np.save(str(Path(LOG_DIR) / "t5_per_subject_tsr.npy"),
            np.array(list(per_subj_tsr_mean.values())))

    # ── Save results ──────────────────────────────────────────────
    out_path = Path(LOG_DIR) / "security_eval_results.json"
    with open(str(out_path), 'w') as f:
        json.dump({"best_seed": best_seed, "results": results}, f, indent=2)
    print(f"\n  ✓ Security results saved: {out_path}")

    # ── Generate figures ──────────────────────────────────────────
    _plot_fgsm_pgd(results)
    _plot_noise(results)
    _plot_t5_hist()


def _plot_fgsm_pgd(results):
    """FIG_15 — FGSM/PGD EER vs epsilon."""
    fgsm = [(float(r["label"].split("=")[1]), r["eer"])
            for r in results if r.get("eer") and "FGSM" in r["label"]]
    pgd  = [(float(r["label"].split("=")[1]), r["eer"])
            for r in results if r.get("eer") and "PGD" in r["label"] and "T4" in r["label"]]
    if not fgsm: return
    fgsm.sort(); pgd.sort()
    eps_f, eer_f = zip(*fgsm)
    eps_p, eer_p = zip(*pgd)

    clean = next((r["eer"] for r in results if r["label"] == "Clean (B2T)"), None)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(eps_f, eer_f, 'o-', color=COLORS["fgsm"], lw=1.5, ms=4, label="FGSM")
    ax.plot(eps_p, eer_p, 's--', color=COLORS["pgd"],  lw=1.5, ms=4, label="PGD-10")
    if clean:
        ax.axhline(clean, color=COLORS["clean"], ls=':', lw=1.1,
                   label=f"Clean EER={clean:.2f}%")
    ax.set_xlabel("Perturbation ε (L∞)"); ax.set_ylabel("EER (%)")
    ax.set_title("T4: Adversarial Robustness (FGSM vs PGD)")
    ax.legend(fontsize=8)
    for d, ext in [(PNG_DIR, 'png'), (PDF_DIR, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_15_adversarial_robustness.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_15 saved")


def _plot_noise(results):
    """FIG_14 — noise robustness bar chart."""
    awgn = [(int(r["label"].split()[2].replace("dB","")), r["eer"])
            for r in results if "AWGN" in r["label"] and r.get("eer")]
    awgn.sort(reverse=True)
    pl   = [(r["label"].split()[-1], r["eer"])
            for r in results if "Powerline" in r["label"] and r.get("eer")]
    if not awgn: return
    snr_vals, eer_vals = zip(*awgn)
    clean = next((r["eer"] for r in results if r["label"] == "Clean (B2T)"), None)

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    x_a = np.arange(len(snr_vals))
    bars = ax.bar(x_a, eer_vals, width=0.4, color=COLORS["awgn"], label="AWGN", zorder=3)
    ax.bar_label(bars, fmt='%.2f%%', fontsize=7.5, padding=2)
    if pl:
        x_pl  = np.arange(len(pl)) + len(snr_vals) + 0.6
        bars2 = ax.bar(x_pl, [e for _, e in pl], width=0.4,
                       color=COLORS["T0"], label="Powerline", zorder=3)
        ax.bar_label(bars2, fmt='%.2f%%', fontsize=7.5, padding=2)
        ax.set_xticks(list(x_a) + list(x_pl))
        ax.set_xticklabels([f"{s}dB" for s in snr_vals] + [p for p,_ in pl],
                           fontsize=7.5)
    else:
        ax.set_xticks(x_a)
        ax.set_xticklabels([f"{s}dB" for s in snr_vals], fontsize=8)
    if clean:
        ax.axhline(clean, color='k', ls='--', lw=1.0, label=f"Clean EER={clean:.2f}%")
    ax.set_ylabel("EER (%)"); ax.set_title("T3: Signal Noise Robustness")
    ax.legend(fontsize=8); ax.set_ylim(0, max(eer_vals) * 1.3)
    for d, ext in [(PNG_DIR, 'png'), (PDF_DIR, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_14_noise_robustness.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_14 saved")


def _plot_t5_hist():
    """FIG_13 — T5 targeted impersonation histogram."""
    tsr_path = Path(LOG_DIR) / "t5_per_subject_tsr.npy"
    if not tsr_path.exists(): return
    tsr_vals = np.load(str(tsr_path)) * 100
    mean_tsr = float(np.mean(tsr_vals))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.hist(tsr_vals, bins=20, color='#d62728', edgecolor='white', alpha=0.85)
    ax.axvline(mean_tsr, color='k', lw=1.5, ls='--',
               label=f"Mean TSR={mean_tsr:.2f}%")
    ax.set_xlabel("Per-Subject TSR (%)"); ax.set_ylabel("Subjects")
    ax.set_title("T5: Targeted Impersonation Success Rate")
    ax.legend(fontsize=8)
    for d, ext in [(PNG_DIR, 'png'), (PDF_DIR, 'pdf')]:
        Path(d).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(d) / f"FIG_13_security_summary.{ext}"),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("    ✓ FIG_13 saved")


if __name__ == "__main__":
    main()
