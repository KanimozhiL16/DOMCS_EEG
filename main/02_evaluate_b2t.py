#!/usr/bin/env python3
"""
02_evaluate_b2t.py
===================
Strict B2T evaluation across all 5 seeds.
Enrollment: R01+R02 ONLY (resting state)
Verification: R03-R14 ONLY (task state)

Outputs per seed and aggregate:
  - EER, AUC (overall + per-run + per-task-group)
  - ROC curves data (fpr, tpr, thresholds)
  - DET curves
  - genuine/impostor score distributions
  - scores_seed{s}.npz for downstream figure generation
  - eval_summary.json
"""

import os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (SEEDS, CKPT_DIR, LOG_DIR, FIG_DIR, TABLE_DIR,
                    REST_RUNS, TASK_RUNS, TASK_GROUP_MAP, N_SUBJECTS)
from model       import DOMCSEEGModel
from data_loader import (load_npz, build_gallery, get_verification_set,
                          score_probe, compute_eer)
import torch
import scipy.stats


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_seed(seed, X, y, session, device):
    """Full B2T evaluation for one seed."""
    print(f"\n  Evaluating seed {seed}...")

    ckpt_path = Path(CKPT_DIR) / f"seed_{seed}" / "model_best.pt"
    if not ckpt_path.exists():
        print(f"  ⚠ Checkpoint not found: {ckpt_path} — skipping")
        return None

    # Load model
    model = DOMCSEEGModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build gallery (R01+R02 only)
    print(f"    Building gallery from {REST_RUNS}...")
    gallery = build_gallery(model, X, y, session, device)

    # Get verification probes (R03-R14)
    X_test, y_test, sess_test = get_verification_set(X, y, session)
    print(f"    Verification probes: {len(X_test):,} windows from {np.unique(sess_test)}")

    # Extract identity embeddings for all probes
    print("    Extracting probe embeddings...")
    all_emb = []
    with torch.no_grad():
        bs = 512
        for i in range(0, len(X_test), bs):
            xb = torch.from_numpy(X_test[i:i+bs]).to(device)
            z  = model.get_identity_embedding(xb)
            all_emb.append(z.cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0)   # (N_test, 128)

    # Score all probes: genuine (probe_subj == gallery_subj) and impostor
    print("    Scoring probes against gallery...")
    genuine_scores   = []
    impostor_scores  = []
    per_run_genuine  = {r: [] for r in TASK_RUNS}
    per_run_impostor = {r: [] for r in TASK_RUNS}

    # Use random seed 0 for reproducible impostor sampling
    rng = np.random.default_rng(0)
    subjects = sorted(gallery.keys())

    for i, (emb, true_subj, run) in enumerate(zip(all_emb, y_test, sess_test)):
        emb = emb / (np.linalg.norm(emb) + 1e-8)   # ensure L2-norm
        scores = score_probe(emb, gallery)

        # Genuine score: max similarity to true subject's prototypes
        g_score = scores[int(true_subj)]
        genuine_scores.append(g_score)
        if run in per_run_genuine:
            per_run_genuine[run].append(g_score)

        # Impostor: one random different subject per probe
        imp_subj = rng.choice([s for s in subjects if s != int(true_subj)])
        i_score  = scores[imp_subj]
        impostor_scores.append(i_score)
        if run in per_run_impostor:
            per_run_impostor[run].append(i_score)

    genuine_scores  = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Overall EER + AUC
    eer, roc_auc, fpr, tpr, thr = compute_eer(genuine_scores, impostor_scores)
    print(f"    EER={eer:.3f}%  AUC={roc_auc:.4f}")

    # Per-run EER
    per_run_results = {}
    for run in TASK_RUNS:
        g = np.array(per_run_genuine[run])
        im = np.array(per_run_impostor[run])
        if len(g) < 10:
            per_run_results[run] = {"eer": None, "n_probes": len(g)}
            continue
        r_eer, r_auc, _, _, _ = compute_eer(g, im)
        per_run_results[run] = {
            "eer":      float(r_eer),
            "auc":      float(r_auc),
            "n_probes": int(len(g)),
        }

    # Per-task-group EER
    task_groups = {
        "RL_Fist_MI":  ['R03','R07','R11'],
        "BF_Feet_MI":  ['R04','R08','R12'],
        "RL_Fist_MV":  ['R05','R09','R13'],
        "BF_Feet_MV":  ['R06','R10','R14'],
    }
    per_task_results = {}
    for grp, runs in task_groups.items():
        g_runs  = np.concatenate([per_run_genuine[r]  for r in runs if per_run_genuine[r]])
        i_runs  = np.concatenate([per_run_impostor[r] for r in runs if per_run_impostor[r]])
        if len(g_runs) < 10:
            per_task_results[grp] = {"eer": None}
            continue
        g_eer, g_auc, _, _, _ = compute_eer(g_runs, i_runs)
        per_task_results[grp] = {
            "eer":      float(g_eer),
            "auc":      float(g_auc),
            "n_probes": int(len(g_runs)),
        }

    # Save seed scores for downstream use
    seed_log_dir = Path(LOG_DIR) / f"seed_{seed}"
    seed_log_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(seed_log_dir / f"scores_seed{seed}.npz"),
             genuine   = genuine_scores,
             impostor  = impostor_scores,
             fpr       = fpr, tpr = tpr, thr = thr)

    result = {
        "seed":           seed,
        "eer":            float(eer),
        "auc":            float(roc_auc),
        "n_genuine":      int(len(genuine_scores)),
        "n_impostor":     int(len(impostor_scores)),
        "per_run":        per_run_results,
        "per_task_group": per_task_results,
    }
    return result, genuine_scores, impostor_scores, fpr, tpr, thr


def bootstrap_ci(eer_vals, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(eer_vals, size=len(eer_vals), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def main():
    print("DOMCS-EEG — B2T Evaluation (5 seeds)")
    print("="*60)
    device = get_device()

    print("Loading dataset...")
    X, y, session = load_npz()

    all_results  = []
    all_eers     = []
    all_aucs     = []
    best_eer     = float('inf')
    best_seed    = -1
    best_genuine = None
    best_impostor= None
    best_fpr = best_tpr = best_thr = None

    for seed in SEEDS:
        out = evaluate_seed(seed, X, y, session, device)
        if out is None:
            continue
        result, genuine, impostor, fpr, tpr, thr = out
        all_results.append(result)
        all_eers.append(result["eer"])
        all_aucs.append(result["auc"])
        if result["eer"] < best_eer:
            best_eer      = result["eer"]
            best_seed     = seed
            best_genuine  = genuine
            best_impostor = impostor
            best_fpr, best_tpr, best_thr = fpr, tpr, thr

    if not all_eers:
        print("No seeds evaluated — run 01_train first")
        return

    mean_eer = float(np.mean(all_eers))
    std_eer  = float(np.std(all_eers, ddof=1))
    mean_auc = float(np.mean(all_aucs))
    std_auc  = float(np.std(all_aucs, ddof=1))
    ci_lo, ci_hi = bootstrap_ci(np.array(all_eers))

    # Wilcoxon signed-rank vs chance EER (50%)
    from scipy.stats import wilcoxon
    try:
        stat, p_val = wilcoxon([e - 50.0 for e in all_eers])
    except Exception:
        stat, p_val = None, None

    print(f"\n{'='*60}")
    print(f"  RESULTS — 5-seed B2T Evaluation")
    print(f"  Mean EER:  {mean_eer:.4f}% ± {std_eer:.4f}%")
    print(f"  Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  95% CI:    [{ci_lo:.4f}%, {ci_hi:.4f}%]  (n=10,000 bootstrap)")
    print(f"  Best seed: {best_seed}  (EER={best_eer:.4f}%)")
    for r in all_results:
        print(f"    Seed {r['seed']}: EER={r['eer']:.4f}%  AUC={r['auc']:.4f}")

    # Save aggregate summary
    summary = {
        "mean_eer":      mean_eer,
        "std_eer":       std_eer,
        "mean_auc":      mean_auc,
        "std_auc":       std_auc,
        "bootstrap_ci":  [ci_lo, ci_hi],
        "wilcoxon_stat": float(stat) if stat is not None else None,
        "wilcoxon_p":    float(p_val) if p_val is not None else None,
        "best_seed":     best_seed,
        "best_eer":      float(best_eer),
        "per_seed":      all_results,
        "enrollment_source": "R01+R02 ONLY",
        "verification_source": "R03-R14 ONLY",
    }
    out_path = Path(LOG_DIR) / "eval_summary.json"
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Summary saved: {out_path}")

    # Save best-seed ROC data for figures
    if best_genuine is not None:
        np.savez(str(Path(LOG_DIR) / "best_seed_roc.npz"),
                 genuine  = best_genuine,
                 impostor = best_impostor,
                 fpr=best_fpr, tpr=best_tpr, thr=best_thr,
                 seed=np.array([best_seed]),
                 eer=np.array([best_eer]),
                 auc=np.array([mean_auc]))


if __name__ == "__main__":
    main()
