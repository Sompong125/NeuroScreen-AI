# ============================================================
# DAY 5.2 - PREPROCESSING TEST (19 CHANNELS)
# Raw → 19 channels → 4s segment → NaN/Inf check
#     → 50 Hz notch filter → standardization → save outputs
# Run: python day5_02_preprocessing_test.py
# ============================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # บันทึกไฟล์โดยไม่เปิดหน้าต่างกราฟ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- settings ----------
NROWS = 50000
FS = 128
SEGMENT_SEC = 4
SEGMENT_SIZE = FS * SEGMENT_SEC  # 512

NOTCH_W0 = 50.0   # ความถี่ power-line (Hz)
NOTCH_Q = 30.0

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
    print("DAY 5.2 - PREPROCESSING TEST (19 CHANNELS)")
    print("=" * 60)

    path = find_dataset()
    if path is None:
        print("ERROR: ไม่พบ adhdata.csv ได้ลองค้นหาที่:")
        for p in CANDIDATES:
            print("  ", p)
        sys.exit(1)

    print(f"\nDataset path: {path}")
    df = pd.read_csv(path, nrows=NROWS)
    df.columns = df.columns.str.strip()

    first_id = df["ID"].iloc[0]
    sub = df[df["ID"] == first_id]
    p_class = sub["Class"].iloc[0]

    if len(sub) < SEGMENT_SIZE:
        print(f"ERROR: participant {first_id} มีเพียง {len(sub)} samples "
              f"ในหน้าต่างที่โหลด (< {SEGMENT_SIZE})")
        sys.exit(1)

    seg = sub.head(SEGMENT_SIZE)
    raw = seg[EEG_CHANNELS].to_numpy(dtype=float)

    print(f"\nParticipant: {first_id}")
    print(f"Class: {p_class}")
    print("Segment shape:", raw.shape, f"(expected ({SEGMENT_SIZE}, 19))")

    nan_count = int(np.isnan(raw).sum())
    inf_count = int(np.isinf(raw).sum())
    print(f"NaN: {nan_count} | Inf: {inf_count}")
    if nan_count or inf_count:
        print("ERROR: พบ NaN/Inf ใน segment — หยุดก่อนกรองสัญญาณ")
        sys.exit(1)

    # ---------- 50 Hz notch filter ----------
    b, a = iirnotch(w0=NOTCH_W0, Q=NOTCH_Q, fs=FS)
    filtered = np.empty_like(raw)
    for i in range(raw.shape[1]):
        filtered[:, i] = filtfilt(b, a, raw[:, i])

    # ---------- standardization ราย channel ----------
    processed = (filtered - filtered.mean(axis=0)) / filtered.std(axis=0)

    orig_mean = raw.mean(axis=0)
    orig_std = raw.std(axis=0)
    proc_mean = processed.mean(axis=0)
    proc_std = processed.std(axis=0)

    print("\n" + "-" * 76)
    print(f"{'Channel':<8}| {'Orig Mean':>12} | {'Orig Std':>10} | "
          f"{'Proc Mean':>12} | {'Proc Std':>10}")
    print("-" * 76)
    for i, ch in enumerate(EEG_CHANNELS):
        print(f"{ch:<8}| {orig_mean[i]:>12.4f} | {orig_std[i]:>10.4f} | "
              f"{proc_mean[i]:>12.2e} | {proc_std[i]:>10.6f}")
    print("-" * 76)

    ok_mean = bool(np.all(np.abs(proc_mean) < 1e-6))
    ok_std = bool(np.all(np.abs(proc_std - 1.0) < 1e-6))
    if ok_mean and ok_std:
        print("\nStandardization PASSED")
    else:
        print("\nStandardization FAILED")
        sys.exit(1)

    # ---------- visualization ----------
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    axes = axes.flatten()
    t = np.arange(SEGMENT_SIZE) / FS
    for i, ch in enumerate(EEG_CHANNELS):
        axes[i].plot(t, processed[:, i], linewidth=0.8, color="#0ea5e9")
        axes[i].set_title(f"{ch} (processed)")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude (SD)")
    axes[-1].axis("off")
    fig.suptitle(
        f"Preprocessed EEG (50 Hz notch + standardization)\n"
        f"Participant: {first_id} | Class: {p_class}", fontsize=16)
    plt.tight_layout()
    png_path = OUT_DIR / "day5_preprocessed_19channels.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")

    # ---------- save CSV (512 rows) ----------
    out_df = pd.DataFrame(processed, columns=EEG_CHANNELS)
    out_df["Class"] = p_class
    out_df["ID"] = first_id
    csv_path = OUT_DIR / "day5_preprocessed_19channels_test.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  rows={len(out_df)}")

    # ---------- summary txt ----------
    lines = [
        "=" * 60,
        "DAY 5.2 - PREPROCESSING SUMMARY",
        "=" * 60,
        f"Dataset path: {path}",
        f"Participant: {first_id}",
        f"Class: {p_class}",
        f"Segment shape: {raw.shape}",
        f"Sampling rate: {FS} Hz (dataset metadata)",
        f"Segment duration: {SEGMENT_SEC} s",
        f"Notch filter: w0={NOTCH_W0} Hz, Q={NOTCH_Q}, fs={FS} (iirnotch + filtfilt)",
        f"NaN: {nan_count} | Inf: {inf_count}",
        "",
        f"{'Channel':<8}| {'Orig Mean':>12} | {'Orig Std':>10} | "
        f"{'Proc Mean':>12} | {'Proc Std':>10}",
    ]
    for i, ch in enumerate(EEG_CHANNELS):
        lines.append(
            f"{ch:<8}| {orig_mean[i]:>12.4f} | {orig_std[i]:>10.4f} | "
            f"{proc_mean[i]:>12.2e} | {proc_std[i]:>10.6f}")
    lines += [
        "",
        f"Processed mean ~ 0: {ok_mean}",
        f"Processed std ~ 1: {ok_std}",
        "Standardization: PASSED" if (ok_mean and ok_std) else "Standardization: FAILED",
        "=" * 60,
    ]
    txt_path = OUT_DIR / "day5_02_preprocessing_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {txt_path}")

    print("\nDay 5.2 PASSED — ไปรัน Day 5.3 ต่อได้")


if __name__ == "__main__":
    main()