# Data Access Instructions

This directory does **not** contain data files (`.npz`, `.edf`). Data files are excluded from the repository because they are large (PHYSIONET: ~1.5 GB preprocessed) and subject to their own usage policies.

---

## PHYSIONET EEGMMIDB (primary dataset)

**Citation:** Schalk G, McFarland DJ, Hinterberger T, Birbaumer N, Wolpaw JR. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Trans Biomed Eng 51(6):1034-1043, 2004.

**Download:**
```
https://physionet.org/content/eegmmidb/1.0.0/
```

**Preprocessing steps to produce `preprocessed_b2t.npz`:**
1. Download all 109 subject `.edf` files (runs R01–R14 per subject)
2. Load with MNE: `mne.io.read_raw_edf()`
3. Bandpass filter: 0.5–45 Hz
4. Resample to 128 Hz
5. Select 64 EEG channels (exclude EOG/EMG if present)
6. Segment into non-overlapping 2-second windows (256 samples), 1-second step
7. Save as `.npz` with keys:
   - `X`: float32 array, shape (N, 64, 256)
   - `y`: int64 array, shape (N,) — subject index 0–108
   - `session`: object array, shape (N,) — run label string, e.g. `'R01'`, `'R14'`

**Expected output:** `preprocessed_b2t.npz` — approximately 173,198 windows total.

Place the file at: `data/preprocessed_b2t.npz` (or pass `--data /your/path/file.npz` to any script).

---

## BED Dataset (cross-dataset validation)

The BED dataset is a 14-channel (Emotiv EPOC+) EEG dataset with 21 subjects and 3 sessions:
- r01: REST (eyes open/closed, Session 1)
- r02: REST (eyes open/closed, Session 2)
- r03: TASK (GAPED/OASIS images + SSVEP stimuli at 3/5/7/10 Hz)

**Access:** Contact the BED dataset authors directly for access. The dataset is not publicly available for automated download.

**Expected file:** `BED_win2s_step1s_fs128.npz` with keys:
- `X`: float32, shape (N, 14, 256)
- `y`: int64, shape (N,) — subject index 0–20
- `session`: object, shape (N,) — session label, e.g. `'r01'`, `'r02'`, `'r03'`
- `stimulus`: object, shape (N,) — stimulus type string

Place the file at: `data/BED_win2s_step1s_fs128.npz` or pass `--data /your/path/file.npz`.

---

## Directory layout after data placement

```
data/
├── README.md                         ← this file
├── preprocessed_b2t.npz              ← PHYSIONET (place here, not tracked by git)
└── BED_win2s_step1s_fs128.npz        ← BED (place here, not tracked by git)
```
