#!/usr/bin/env python3
"""
10_protocol_comparison.py — DOMCS-EEG Protocol Comparison
===========================================================
Evaluates the SAME trained model (seed=3 checkpoint) under three protocols
to fill TABLE_03 in the paper.

Protocols:
  P1: B2T (Enroll R01+R02, Verify R03-R14) — THE MAIN PROTOCOL
      Already computed in eval_summary.json. Loaded directly, not re-computed.
  P2: Same-session 80/20 split (train and test on SAME sessions)
      Uses seed_3 checkpoint. Gallery from 80% of each session, probe from 20%.
  P3: Random-window 80/20 split (ignore session boundaries)
      Uses seed_3 checkpoint. Gallery from random 80% of ALL windows, probe from 20%.

CRITICAL: Run AFTER 01_train has finished (checkpoint must exist).
  cd /home/nvidia/24PHD1237/DOMCS_EEG_FINAL_LOCKED_TIFS_VERSION/code
  python 10_protocol_comparison.py

Outputs:
  logs/protocol_comparison.json
  tables/TABLE_03_protocol_comparison.csv
  latex_tables/TABLE_03_protocol_comparison.tex

NOTE on interpretation:
  P2 and P3 will show LOWER EER than P1 because enrollment and verification
  windows come from the same session/distribution — no cross-state generalisation.
  The paper argues P1 (B2T) is the scientifically correct metric.
"""

import os, sys, json, csv
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (LOG_DIR, CKPT_DIR, TABLE_DIR, LATEX_DIR, NPZ_PATH,
                    REST_RUNS, TASK_RUNS, N_SUBJECTS, GALLERY_K)
from model       import DOMCSEEGModel
from data_loader import (load_npz, build_gallery, get_verification_set,
                          score_probe, compute_eer)
from sklearn.cluster import KMeans

EVAL_SEED = 3


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(seed, device):
    ckpt_path = Path(CKPT_DIR) / f"seed_{seed}" / "model_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun 01_train first.")
    model = DOMCSEEGModel().to(device)
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


def extract_embeddings(model, X, device, batch_size=512):
    """Extract z_id for all windows in X."""
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            z  = model.get_identity_embedding(xb)
            all_emb.append(z.cpu().numpy())
    return np.concatenate(all_emb, axis=0)   # (N, 128)


def build_kmeans_gallery(emb, y_subj, k=GALLERY_K):
    """Build KMeans gallery from embeddings. Returns {subj: (k, 128)}."""
    gallery = {}
    for subj in np.unique(y_subj):
        mask   = y_subj == subj
        emb_s  = emb[mask]
        k_s    = min(k, len(emb_s))
        km     = KMeans(n_clusters=k_s, n_init=10, random_state=42)
        km.fit(emb_s)
        protos = km.cluster_centers_
        norms  = np.linalg.norm(protos, axis=1, keepdims=True)
        gallery[int(subj)] = protos / (norms + 1e-8)
    return gallery


def run_verification(emb_probe, y_probe, gallery):
    """Compute genuine/impostor scores. Returns (eer, auc)."""
    genuine, impostor = [], []
    subjects = np.unique(y_probe)
    for subj in subjects:
        mask = y_probe == subj
        for p in emb_probe[mask]:
            # L2-normalise probe
            p_n = p / (np.linalg.norm(p) + 1e-8)
            scores = score_probe(p_n, gallery)
            genuine.append(scores.get(int(subj), 0.0))
            for k, v in scores.items():
                if k != int(subj):
                    impostor.append(v)
    eer, roc_auc, _, _, _ = compute_eer(np.array(genuine), np.array(impostor))
    return float(eer), float(roc_auc)


# ─── Protocol P1: B2T (from saved results) ────────────────────────────────────

def protocol_p1_b2t():
    """Load B2T EER/AUC from saved eval_summary.json (seed=3)."""
    print("\n  P1: B2T — loading from eval_summary.json...")
    json_path = Path(LOG_DIR) / "eval_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found. Run 02_evaluate_b2t first.")
    with open(str(json_path)) as f:
        summary = json.load(f)

    best_seed = summary.get("best_seed", EVAL_SEED)
    eer_by_seed = summary.get("eer_by_seed", {})
    auc_by_seed = summary.get("auc_by_seed", {})

    eer = eer_by_seed.get(str(EVAL_SEED), summary.get("mean_eer"))
    auc = auc_by_seed.get(str(EVAL_SEED), summary.get("mean_auc"))

    print(f"  P1 (B2T, seed={EVAL_SEED}): EER={eer:.4f}%  AUC={auc:.4f}")
    return float(eer), float(auc)


# ─── Protocol P2: Same-session 80/20 ──────────────────────────────────────────

def protocol_p2_same_session(model, X, y, session, device):
    """
    Same-session evaluation: for EACH run, 80% windows → gallery, 20% → probes.
    Evaluate across all runs (R01-R14) combined.
    """
    print("\n  P2: Same-session 80/20 split...")
    rng = np.random.default_rng(EVAL_SEED)

    all_sessions = np.unique(session)
    enroll_X, enroll_y, probe_X, probe_y = [], [], [], []

    for run in all_sessions:
        mask = session == run
        X_run = X[mask]; y_run = y[mask]
        idx   = rng.permutation(len(X_run))
        n_enroll = int(0.8 * len(idx))
        enroll_X.append(X_run[idx[:n_enroll]])
        enroll_y.append(y_run[idx[:n_enroll]])
        probe_X.append(X_run[idx[n_enroll:]])
        probe_y.append(y_run[idx[n_enroll:]])

    enroll_X = np.concatenate(enroll_X)
    enroll_y = np.concatenate(enroll_y)
    probe_X  = np.concatenate(probe_X)
    probe_y  = np.concatenate(probe_y)

    print(f"    Enrollment: {len(enroll_X):,} windows  |  Probe: {len(probe_X):,} windows")

    emb_enroll = extract_embeddings(model, enroll_X, device)
    emb_probe  = extract_embeddings(model, probe_X,  device)

    gallery = build_kmeans_gallery(emb_enroll, enroll_y)
    eer, auc = run_verification(emb_probe, probe_y, gallery)
    print(f"  P2 (same-session): EER={eer:.4f}%  AUC={auc:.4f}")
    return eer, auc


# ─── Protocol P3: Random 80/20 ────────────────────────────────────────────────

def protocol_p3_random_split(model, X, y, device):
    """
    Random-window 80/20 split ignoring session boundaries.
    80% of all windows → gallery, 20% → probes.
    """
    print("\n  P3: Random 80/20 split (session-agnostic)...")
    rng = np.random.default_rng(EVAL_SEED)
    idx      = rng.permutation(len(X))
    n_enroll = int(0.8 * len(idx))
    enroll_X = X[idx[:n_enroll]];  enroll_y = y[idx[:n_enroll]]
    probe_X  = X[idx[n_enroll:]];  probe_y  = y[idx[n_enroll:]]

    print(f"    Enrollment: {len(enroll_X):,} windows  |  Probe: {len(probe_X):,} windows")

    emb_enroll = extract_embeddings(model, enroll_X, device)
    emb_probe  = extract_embeddings(model, probe_X,  device)

    gallery = build_kmeans_gallery(emb_enroll, enroll_y)
    eer, auc = run_verification(emb_probe, probe_y, gallery)
    print(f"  P3 (random-split): EER={eer:.4f}%  AUC={auc:.4f}")
    return eer, auc


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(p1_eer, p1_auc, p2_eer, p2_auc, p3_eer, p3_auc):
    results = {
        "seed": EVAL_SEED,
        "P1_B2T": {"description": "Enroll R01+R02, Verify R03-R14 (strict cross-state)",
                   "eer": p1_eer, "auc": p1_auc},
        "P2_same_session": {"description": "Same-session 80/20 split",
                            "eer": p2_eer, "auc": p2_auc},
        "P3_random_split": {"description": "Random 80/20 split (session-agnostic)",
                            "eer": p3_eer, "auc": p3_auc},
    }

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    json_path = Path(LOG_DIR) / "protocol_comparison.json"
    with open(str(json_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ JSON: {json_path}")

    # CSV
    Path(TABLE_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = Path(TABLE_DIR) / "TABLE_03_protocol_comparison.csv"
    rows = [
        {"Protocol": "P1 B2T (proposed)",
         "Description": "Enroll R01+R02 (rest), Verify R03-R14 (task) — strict cross-state",
         "EER (%)": f"{p1_eer:.4f}", "AUC": f"{p1_auc:.4f}"},
        {"Protocol": "P2 Same-session 80/20",
         "Description": "80% enroll / 20% probe, same session distribution",
         "EER (%)": f"{p2_eer:.4f}", "AUC": f"{p2_auc:.4f}"},
        {"Protocol": "P3 Random 80/20",
         "Description": "Random window split, ignores session boundaries",
         "EER (%)": f"{p3_eer:.4f}", "AUC": f"{p3_auc:.4f}"},
    ]
    with open(str(csv_path), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["Protocol","Description","EER (%)","AUC"])
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓ CSV: {csv_path}")

    # LaTeX
    Path(LATEX_DIR).mkdir(parents=True, exist_ok=True)
    tex_path = Path(LATEX_DIR) / "TABLE_03_protocol_comparison.tex"
    with open(str(tex_path), 'w') as f:
        f.write(r"""\begin{table}[!t]
\centering
\caption{Protocol comparison — EER (\%) under three evaluation settings
(seed=3 checkpoint). P1 (B2T) is the proposed protocol; P2 and P3 show
overly optimistic EER because enrollment and probe windows share the same
session or distribution, violating the cross-state generalisation requirement.}
\label{tab:protocol}
\begin{tabular}{lrr}
\hline
\textbf{Protocol} & \textbf{EER (\%)} & \textbf{AUC} \\
\hline
""")
        f.write(f"P1: B2T — Enroll $R_{{01}}$+$R_{{02}}$, Verify $R_{{03}}$--$R_{{14}}$ (proposed) "
                f"& {p1_eer:.4f} & {p1_auc:.4f} \\\\\n")
        f.write(f"P2: Same-session 80/20 split & {p2_eer:.4f} & {p2_auc:.4f} \\\\\n")
        f.write(f"P3: Random 80/20 split (session-agnostic) & {p3_eer:.4f} & {p3_auc:.4f} \\\\\n")
        f.write(r"""\hline
\end{tabular}
\end{table}
""")
    print(f"  ✓ LaTeX: {tex_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DOMCS-EEG Protocol Comparison (seed=3)")
    print("=" * 60)

    device = get_device()

    print("\nLoading data...")
    X, y, session = load_npz(NPZ_PATH)

    # Load model once — used for P2 and P3
    model = load_model(EVAL_SEED, device)

    # Run protocols
    p1_eer, p1_auc = protocol_p1_b2t()
    p2_eer, p2_auc = protocol_p2_same_session(model, X, y, session, device)
    p3_eer, p3_auc = protocol_p3_random_split(model, X, y, device)

    print("\n" + "=" * 60)
    print("PROTOCOL COMPARISON RESULTS")
    print("=" * 60)
    print(f"  P1 B2T (proposed):    EER={p1_eer:.4f}%  AUC={p1_auc:.4f}")
    print(f"  P2 Same-session:      EER={p2_eer:.4f}%  AUC={p2_auc:.4f}")
    print(f"  P3 Random-split:      EER={p3_eer:.4f}%  AUC={p3_auc:.4f}")
    print(f"\n  Note: P2/P3 lower EER is EXPECTED — they violate cross-state separation.")
    print(f"  B2T protocol (P1) is the scientifically correct metric.")

    save_results(p1_eer, p1_auc, p2_eer, p2_auc, p3_eer, p3_auc)
    print("\n✓ Protocol comparison complete. Copy TABLE_03 to paper.")


if __name__ == "__main__":
    main()
