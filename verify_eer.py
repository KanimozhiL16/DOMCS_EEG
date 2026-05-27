"""
DOMCS-EEG — EER Verification from Pre-computed Results
=======================================================
Verifies the reported EER values directly from eval_summary.json
(no GPU, no dataset download, no training required).

Usage:
    python verify_eer.py

Requires only:
    checkpoints/eval_summary.json  (included in this repo)
"""

import json
import sys
import statistics
from pathlib import Path

EXPECTED = {
    1: {"eer": 2.4978, "auc": 0.9962},
    2: {"eer": 2.4587, "auc": 0.9955},
    3: {"eer": 2.2823, "auc": 0.9962},
    4: {"eer": 2.3582, "auc": 0.9967},
    5: {"eer": 2.4718, "auc": 0.9963},
}
EXPECTED_MEAN = 2.4138
EXPECTED_STD  = 0.0906
TOLERANCE_EER = 0.001   # % — exact match expected (stored values)


def main():
    summary_path = Path(__file__).parent / "checkpoints" / "eval_summary.json"
    if not summary_path.exists():
        print(f"[ERROR] {summary_path} not found.")
        print("  See README.md — checkpoints/ must be present.")
        sys.exit(1)

    with open(summary_path) as f:
        results = json.load(f)

    print("=" * 60)
    print("  DOMCS-EEG EER Verification")
    print("  (no GPU or dataset required)")
    print("=" * 60)
    print(f"\n  {'Seed':<10} {'EER (paper)':<16} {'EER (stored)':<16} {'Match'}")
    print(f"  {'-'*52}")

    eers = []
    all_pass = True
    per_seed = results.get("per_seed", [])

    for entry in per_seed:
        seed       = entry["seed"]
        stored_eer = entry.get("eer_pct", entry.get("eer"))
        exp        = EXPECTED.get(seed)

        if exp is None or stored_eer is None:
            print(f"  seed_{seed:<6}  MISSING / UNEXPECTED SEED")
            all_pass = False
            continue

        delta = abs(stored_eer - exp["eer"])
        ok    = delta <= TOLERANCE_EER
        if not ok:
            all_pass = False
        eers.append(stored_eer)
        print(f"  seed_{seed:<6} {exp['eer']:.4f}%          {stored_eer:.4f}%          {'✅' if ok else '❌'}")

    if eers:
        mean = statistics.mean(eers)
        std  = statistics.stdev(eers) if len(eers) > 1 else 0.0
        mean_ok = abs(mean - EXPECTED_MEAN) < 0.001
        print(f"\n  {'Mean±SD':<10} {EXPECTED_MEAN:.4f}%±{EXPECTED_STD:.4f}%  "
              f"{mean:.4f}%±{std:.4f}%  {'✅' if mean_ok else '❌'}")

    print(f"\n{'='*60}")
    if all_pass:
        print("  ✅ All EER values verified. Results match the paper.")
    else:
        print("  ❌ Mismatch detected. Check checkpoints/eval_summary.json.")
    print("=" * 60)
    print("""
To verify from scratch (requires PHYSIONET data + GPU):
  cd main/
  python 02_evaluate_b2t.py --ckpt-dir ../checkpoints/

To verify architecture only (no data, no GPU):
  cd main/
  python model.py
  → Expected: Total params: 234,880
  → Expected: ✓ State branch detach verified
""")


if __name__ == "__main__":
    main()
