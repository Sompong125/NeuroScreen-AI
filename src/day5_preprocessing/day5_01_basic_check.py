# ============================================================
# DAY 5.1 - BASIC EEG CHECK
# โหลดเฉพาะ 50,000 rows เพื่อตรวจสอบโครงสร้างและ participant คนแรก
# Run: python day5_01_basic_check.py
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- settings ----------
NROWS = 50000
FS = 128
SEGMENT_SEC = 4
REQUIRED_SAMPLES = FS * SEGMENT_SEC  # 512

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

CANDIDATES = [
    BASE_DIR.parent / "data" / "adhdata.csv",
    BASE_DIR.parent / "adhdata.csv",
    BASE_DIR.parent.parent / "data" / "adhdata.csv",
    BASE_DIR / "data" / "adhdata.csv",
]


def find_dataset():
    for p in CANDIDATES:
        if p.exists():
            return p
    return None


def main():
    print("=" * 60)
    print("DAY 5.1 - BASIC EEG CHECK")
    print("=" * 60)

    path = find_dataset()
    if path is None:
        msg = "ERROR: ไม่พบ adhdata.csv ได้ลองค้นหาที่:\n  " + \
              "\n  ".join(str(p) for p in CANDIDATES)
        print(msg)
        (OUT_DIR / "day5_01_basic_check.txt").write_text(
            msg + "\nStatus: FAILED\n", encoding="utf-8")
        sys.exit(1)

    print(f"\nDataset path: {path}")
    df = pd.read_csv(path, nrows=NROWS)
    df.columns = df.columns.str.strip()

    print("\nDataset shape (loaded):", df.shape)

    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    missing_meta = [c for c in ("Class", "ID") if c not in df.columns]

    first_id = df["ID"].iloc[0]
    sub = df[df["ID"] == first_id]
    n_samples = len(sub)
    p_class = sub["Class"].iloc[0]

    if missing:
        nan_count = inf_count = -1
        eeg = None
    else:
        eeg = sub[EEG_CHANNELS].to_numpy(dtype=float)
        nan_count = int(np.isnan(eeg).sum())
        inf_count = int(np.isinf(eeg).sum())

    ok_channels = (not missing) and (not missing_meta)
    ok_samples = n_samples >= REQUIRED_SAMPLES
    ok_values = (nan_count == 0) and (inf_count == 0)
    status = "PASSED" if (ok_channels and ok_samples and ok_values) else "FAILED"

    print("\nParticipant:")
    print(first_id)
    print("\nClass:")
    print(p_class)
    print("\nSamples:")
    print(n_samples)
    print("\nEEG Channels:")
    print(len(EEG_CHANNELS) - len(missing))
    if missing:
        print("Missing channels:", missing)
    print("\nNaN:")
    print(nan_count)
    print("\nInf:")
    print(inf_count)
    print("\nRequired samples:")
    print(REQUIRED_SAMPLES)
    print("\nStatus:")
    print(status)
    print("=" * 60)

    lines = [
        "=" * 60,
        "DAY 5.1 - BASIC EEG CHECK",
        "=" * 60,
        f"Dataset path: {path}",
        f"Dataset shape (loaded): {df.shape}",
        f"Participant: {first_id}",
        f"Class: {p_class}",
        f"Samples: {n_samples}",
        f"EEG Channels: {len(EEG_CHANNELS) - len(missing)}/{len(EEG_CHANNELS)}",
        f"Missing channels: {missing if missing else 'none'}",
        f"NaN: {nan_count}",
        f"Inf: {inf_count}",
        f"Required samples: {REQUIRED_SAMPLES}",
        f"Status: {status}",
        "=" * 60,
    ]
    out_txt = OUT_DIR / "day5_01_basic_check.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_txt}")

    if status != "PASSED":
        print("\nDay 5.1 FAILED — หยุดที่นี่ แก้ปัญหาก่อนรัน Day 5.2")
        sys.exit(1)
    print("\nDay 5.1 PASSED — ไปรัน Day 5.2 ต่อได้")


if __name__ == "__main__":
    main()