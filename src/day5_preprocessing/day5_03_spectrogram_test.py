# ============================================================
# DAY 5.3 - SPECTROGRAM TEST
# โหลดเฉพาะ output/day5_preprocessed_19channels_test.csv
# สร้าง spectrogram ทั้ง 19 channels แล้วบันทึกภาพ + สรุปผล
# Run: python day5_03_spectrogram_test.py
# ============================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"

# ---------- settings ----------
FS = 128
NPERSEG = 128
NOVERLAP = 64

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]


def main():
    print("=" * 60)
    print("DAY 5.3 - SPECTROGRAM TEST")
    print("=" * 60)

    csv_path = OUT_DIR / "day5_preprocessed_19channels_test.csv"
    if not csv_path.exists():
        print(f"ERROR: ไม่พบไฟล์ {csv_path}")
        print("       กรุณารัน day5_02_preprocessing_test.py ก่อน")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    participant = df["ID"].iloc[0]
    p_class = df["Class"].iloc[0]
    eeg = df[EEG_CHANNELS].to_numpy(dtype=float)

    print(f"\nParticipant: {participant}")
    print(f"Class: {p_class}")
    print("Input shape:", eeg.shape)
    print("Number of channels:", len(EEG_CHANNELS))
    print(f"Sampling rate: {FS} Hz")
    print(f"Segment duration: {len(eeg) / FS:.2f} s")

    # ---------- สร้าง spectrogram ----------
    specs = {}
    for i, ch in enumerate(EEG_CHANNELS):
        f, t, S = spectrogram(
            eeg[:, i], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
            scaling="density", mode="psd",
        )
        specs[ch] = (f, t, 10.0 * np.log10(S + 1e-12))

    vmin = min(v[2].min() for v in specs.values())
    vmax = max(v[2].max() for v in specs.values())

    # ---------- visualization ----------
    fig, axes = plt.subplots(5, 4, figsize=(18, 20))
    axes = axes.flatten()
    for i, ch in enumerate(EEG_CHANNELS):
        f, t, db = specs[ch]
        im = axes[i].pcolormesh(
            t, f, db, shading="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i].set_title(ch)
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Frequency (Hz)")
        axes[i].set_ylim(0, 60)
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="Power (dB)")
    fig.suptitle(
        f"EEG Spectrogram - 19 Channels (after preprocessing)\n"
        f"Participant: {participant} | Class: {p_class}", fontsize=16)
    plt.tight_layout()
    png_path = OUT_DIR / "day5_spectrogram_19channels.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {png_path}")

    # ---------- summary ----------
    f0, t0, db0 = specs["Fp1"]
    lines = [
        "=" * 60,
        "DAY 5.3 - SPECTROGRAM SUMMARY",
        "=" * 60,
        f"Participant: {participant}",
        f"Class: {p_class}",
        f"Input shape: {eeg.shape}",
        f"Number of channels: {len(EEG_CHANNELS)}",
        f"Sampling rate: {FS} Hz (dataset metadata)",
        f"Segment duration: {len(eeg) / FS:.2f} s",
        f"Spectrogram settings: nperseg={NPERSEG}, noverlap={NOVERLAP}",
        f"Frequency bins: {len(f0)}",
        f"Time bins: {len(t0)}",
        f"Spectrogram shape (per channel): {db0.shape}",
        f"Frequency display range: 0-60 Hz",
        "=" * 60,
    ]
    txt_path = OUT_DIR / "day5_03_spectrogram_summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {txt_path}")

    print("\n" + "=" * 60)
    print("SPECTROGRAM INFORMATION (per channel)")
    print("=" * 60)
    for ch in EEG_CHANNELS:
        f, t, db = specs[ch]
        print(f"{ch:<5} freq bins: {len(f):<4} time bins: {len(t):<4} "
              f"shape: {db.shape}")

    print("\nDay 5.3 PASSED — preprocessing pipeline พร้อมสำหรับขั้นต่อไป")


if __name__ == "__main__":
    main()