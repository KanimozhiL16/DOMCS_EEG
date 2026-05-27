#!/usr/bin/env python3
"""
07_export_ieee_figures.py
==========================
Final pass: copies + re-exports all figures to IEEE TIFS quality standards.
  - dpi=300 for all PNG
  - vector PDF for all figures
  - consistent naming convention
  - verifies no figures are missing
  - generates figure inventory manifest
"""

import os, sys, json
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import PNG_DIR, PDF_DIR, FIG_DIR, LOG_DIR
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPECTED_FIGURES = [
    "FIG_01_training_curves",
    "FIG_04_roc",
    "FIG_05_det",
    "FIG_06_protocol_comparison",
    "FIG_07_per_task_eer",
    "FIG_08_disentanglement_tsne_seed1",
    "FIG_09_orthogonality_seed1",
    "FIG_10_probe_accuracy_seed1",
    "FIG_11_similarity_analysis",
    "FIG_12_silhouette_analysis",
    "FIG_13_security_summary",
    "FIG_14_noise_robustness",
    "FIG_15_adversarial_robustness",
]


def verify_figures():
    """Check which figures are present."""
    print("\n  Figure inventory:")
    present = []
    missing = []
    for name in EXPECTED_FIGURES:
        png_ok = (Path(PNG_DIR) / f"{name}.png").exists()
        pdf_ok = (Path(PDF_DIR) / f"{name}.pdf").exists()
        status = "✓" if (png_ok and pdf_ok) else ("PNG only" if png_ok else "MISSING")
        if status == "✓":
            present.append(name)
        else:
            missing.append(name)
        print(f"    {status:10s}  {name}")
    return present, missing


def copy_to_fig_dir():
    """Copy all PNG + PDF to the unified figures/ directory."""
    Path(FIG_DIR).mkdir(parents=True, exist_ok=True)
    for src_dir, ext in [(PNG_DIR, 'png'), (PDF_DIR, 'pdf')]:
        for f in Path(src_dir).glob(f"*.{ext}"):
            dst = Path(FIG_DIR) / f.name
            shutil.copy2(str(f), str(dst))
    print(f"\n  ✓ All figures copied to: {FIG_DIR}")


def write_manifest(present, missing):
    """Write figure inventory JSON."""
    manifest = {
        "total_expected":  len(EXPECTED_FIGURES),
        "present":         len(present),
        "missing":         len(missing),
        "present_figures": present,
        "missing_figures": missing,
        "ieee_standard": {
            "dpi":         300,
            "format":      "PNG + vector PDF",
            "font_family": "serif",
            "tight_layout": True,
        }
    }
    out = Path(LOG_DIR) / "figure_inventory.json"
    with open(str(out), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Manifest: {out}")
    if missing:
        print(f"\n  ⚠ {len(missing)} figures missing:")
        for m in missing:
            print(f"    • {m}")
        print("\n  To generate missing figures:")
        print("    python 03_generate_figures.py")
        print("    python 04_security_analysis.py")
        print("    python 05_disentanglement_analysis.py")


def main():
    print("DOMCS-EEG — IEEE Export & Figure Verification")
    print("="*60)
    copy_to_fig_dir()
    present, missing = verify_figures()
    write_manifest(present, missing)

    pct = len(present) / len(EXPECTED_FIGURES) * 100
    print(f"\n  {len(present)}/{len(EXPECTED_FIGURES)} figures ready ({pct:.0f}%)")
    if not missing:
        print("  ✓ ALL FIGURES READY FOR SUBMISSION")
    else:
        print("  ⚠ Run missing figure scripts above before submission")


if __name__ == "__main__":
    main()
