# ============================================================
# DAY 7.1 - CNN INPUT DATASET TEST
# Raw EEG Segment → 50 Hz Notch → Standardization → Spectrogram
# → CNN-ready input (4, 19, 65, 7)
#
# ทดสอบเพียง 4 segments (Train 2 / Validation 1 / Test 1)
# ยังไม่ train CNN / ไม่สร้าง model / ไม่ evaluate
# Run: python day7_01_cnn_input_test.py
# ============================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch, spectrogram

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CANDIDATES = [
    BASE_DIR.parent / "data" / "adhdata.csv",
    BASE_DIR.parent / "adhdata.csv",
    BASE_DIR.parent.parent / "data" / "adhdata.csv",
    BASE_DIR / "data" / "adhdata.csv",
]

MANIFEST_CANDIDATES = [
    BASE_DIR.parent / "day6_split" / "output",
    BASE_DIR / "day6_split" / "output",
    BASE_DIR.parent.parent / "day6_split" / "output",
]

# ---------- settings ----------
FS = 128
SEGMENT_SIZE = 512
SEGMENT_SEC = 4
NPERSEG = 128
NOVERLAP = 64
NOTCH_W0 = 50.0
NOTCH_Q = 30.0
CHUNK = 200_000

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
CLASS_TO_LABEL = {"ADHD": 1, "Control": 0}


def find_first(candidates, name):
    for p in candidates:
        if (p / name).exists() if p.is_dir() else p.exists():
            return p if not p.is_dir() else p / name
    return None


def find_dataset():
    for p in DATASET_CANDIDATES:
        if p.exists():
            return p
    return None


def find_manifest_dir():
    for d in MANIFEST_CANDIDATES:
        if d.exists() and (d / "segment_manifest.csv").exists():
            return d
    return None


def select_segments(manifest):
    """เลือก Train 2 segments (คนละ participant), Validation 1, Test 1"""
    chosen = []
    for split, n in [("Train", 2), ("Validation", 1), ("Test", 1)]:
        sub = manifest[manifest["Split"] == split]
        seen = set()
        for _, row in sub.iterrows():
            if row["ID"] in seen:
                continue
            seen.add(row["ID"])
            chosen.append(row.to_dict())
            if len(seen) >= n:
                break
    return chosen


def extract_segments(dataset_path, requests):
    """ดึงเฉพาะ rows ของ segments ที่ต้องการ แบบ chunked (memory-efficient)"""
    need = {r["ID"]: r for r in requests}
    counters = {pid: 0 for pid in need}
    collected = {pid: [] for pid in need}
    done = set()
    cols = ["ID"] + EEG_CHANNELS

    for chunk in pd.read_csv(dataset_path, usecols=cols, chunksize=CHUNK):
        for pid in list(need.keys()):
            if pid in done:
                continue
            r = need[pid]
            sub = chunk[chunk["ID"] == pid]
            base = counters[pid]
            if len(sub) > 0:
                lo_glob = max(r["Start_Sample"], base)
                hi_glob = min(r["End_Sample"], base + len(sub) - 1)
                if lo_glob <= hi_glob:
                    lo = lo_glob - base
                    hi = hi_glob - base
                    collected[pid].append(
                        sub.iloc[lo:hi + 1][EEG_CHANNELS].to_numpy(dtype=float))
            counters[pid] = base + len(sub)
            if counters[pid] > r["End_Sample"]:
                done.add(pid)
        if len(done) == len(need):
            break

    out = {}
    for pid in need:
        if collected[pid]:
            out[pid] = np.concatenate(collected[pid], axis=0)
        else:
            out[pid] = np.empty((0, len(EEG_CHANNELS)))
    return out


def preprocess(raw):
    """NaN/Inf check → 50 Hz notch → standardization (แบบ Day 5)"""
    nan_n = int(np.isnan(raw).sum())
    inf_n = int(np.isinf(raw).sum())
    if nan_n or inf_n:
        return None, nan_n, inf_n
    b, a = iirnotch(w0=NOTCH_W0, Q=NOTCH_Q, fs=FS)
    filtered = np.empty_like(raw)
    for i in range(raw.shape[1]):
        filtered[:, i] = filtfilt(b, a, raw[:, i])
    processed = (filtered - filtered.mean(axis=0)) / filtered.std(axis=0)
    return processed, nan_n, inf_n


def make_spectrogram(processed):
    """spectrogram ทุก channel → (19, 65, 7) ในหน่วย dB
    เก็บครบ 65 bins (0-64 Hz) — การจำกัด 0-60 Hz ทำเฉพาะตอน plot ด้วย set_ylim"""
    specs = []
    for i in range(processed.shape[1]):
        f, t, S = spectrogram(
            processed[:, i], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
            scaling="density", mode="psd")
        specs.append(10.0 * np.log10(S + 1e-12))
    return np.stack(specs, axis=0), f, t


def main():
    print("=" * 60)
    print("DAY 7.1 - CNN INPUT DATASET TEST")
    print("=" * 60)

    dataset_path = find_dataset()
    manifest_dir = find_manifest_dir()
    if dataset_path is None or manifest_dir is None:
        print("ERROR: ไม่พบ adhdata.csv หรือ day6_split/output/")
        sys.exit(1)
    print(f"\nDataset: {dataset_path}")
    print(f"Manifest dir: {manifest_dir}")

    manifest = pd.read_csv(manifest_dir / "segment_manifest.csv")
    split_ids = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_csv(manifest_dir / f"{name}_participants.csv")
        split_ids[name.capitalize()] = set(df["ID"])

    # ---------- เลือก segments ----------
    chosen = select_segments(manifest)
    if len(chosen) != 4:
        print("ERROR: เลือก segments ไม่ได้ครบ 4")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("TEST SEGMENTS")
    print("-" * 60)
    n_tr = sum(1 for c in chosen if c["Split"] == "Train")
    n_va = sum(1 for c in chosen if c["Split"] == "Validation")
    n_te = sum(1 for c in chosen if c["Split"] == "Test")
    print("\nTrain:")
    print(n_tr)
    print("\nValidation:")
    print(n_va)
    print("\nTest:")
    print(n_te)
    print("\nTotal:")
    print(len(chosen))
    print("\nSelected segments:")
    for c in chosen:
        print(f"  {c['Segment_ID']}  | {c['ID']} | {c['Class']} | "
              f"{c['Split']} | start={c['Start_Sample']} end={c['End_Sample']}")

    # ---------- ดึงข้อมูลจริง ----------
    raws = extract_segments(dataset_path, chosen)

    raw_ok = True
    prep_ok = True
    spec_ok = True
    rows_summary = []
    cnn_list = []
    spec_example = None

    for c in chosen:
        raw = raws[c["ID"]]
        if raw.shape != (SEGMENT_SIZE, len(EEG_CHANNELS)):
            print(f"ERROR: segment {c['Segment_ID']} ได้ shape {raw.shape}")
            raw_ok = False
            continue
        if not (np.isfinite(raw).all()):
            raw_ok = False

        processed, nan_n, inf_n = preprocess(raw)
        if processed is None:
            prep_ok = False
            print(f"ERROR: segment {c['Segment_ID']} มี NaN={nan_n} Inf={inf_n}")
            continue
        ok_mean = bool(np.all(np.abs(processed.mean(axis=0)) < 1e-6))
        ok_std = bool(np.all(np.abs(processed.std(axis=0) - 1.0) < 1e-6))
        if not (ok_mean and ok_std):
            prep_ok = False

        spec, freqs, times = make_spectrogram(processed)
        if spec.shape != (len(EEG_CHANNELS), 65, 7) or not np.isfinite(spec).all():
            spec_ok = False
        if spec_example is None:
            spec_example = (c, spec, freqs, times)

        cnn_list.append(spec)
        rows_summary.append({
            "Segment_ID": c["Segment_ID"],
            "ID": c["ID"],
            "Class": c["Class"],
            "Split": c["Split"],
            "Raw_Shape": str(raw.shape),
            "Preprocessed_Shape": str(processed.shape),
            "Spectrogram_Shape": str(spec.shape[1:]),
            "Min_Value": round(float(processed.min()), 6),
            "Max_Value": round(float(processed.max()), 6),
            "Mean_Value": round(float(processed.mean()), 6),
            "Std_Value": round(float(processed.std()), 6),
        })

    cnn_input = np.stack(cnn_list, axis=0) if cnn_list else np.empty((0,))
    cnn_ok = cnn_input.shape == (4, len(EEG_CHANNELS), 65, 7) and \
        bool(np.isfinite(cnn_input).all())

    # ---------- labels ----------
    labels = []
    for i, c in enumerate(chosen):
        labels.append({
            "Index": i,
            "Segment_ID": c["Segment_ID"],
            "ID": c["ID"],
            "Class": c["Class"],
            "Split": c["Split"],
            "Label": CLASS_TO_LABEL[c["Class"]],
        })
    label_df = pd.DataFrame(labels)
    label_ok = set(label_df["Label"]) <= {0, 1} and \
        label_df["Segment_ID"].is_unique and len(label_df) == 4

    # ---------- leakage ----------
    leak = (len(split_ids["Train"] & split_ids["Validation"])
            + len(split_ids["Train"] & split_ids["Test"])
            + len(split_ids["Validation"] & split_ids["Test"]))
    multi = manifest.groupby("ID")["Split"].nunique()
    leak += int((multi > 1).sum())
    for c in chosen:
        if c["ID"] not in split_ids[c["Split"]]:
            leak += 1
    test_pids = [c["ID"] for c in chosen if c["Split"] == "Test"]
    for pid in test_pids:
        if pid in split_ids["Train"] or pid in split_ids["Validation"]:
            leak += 1

    if leak > 0:
        print("\nDATA LEAKAGE DETECTED")
        sys.exit(1)

    # ---------- visualization ----------
    if spec_example is not None:
        c, spec, freqs, times = spec_example
        show_ch = ["Fp1", "Fp2", "F3", "F4"]
        idxs = [EEG_CHANNELS.index(ch) for ch in show_ch]
        vmin = min(spec[i].min() for i in idxs)
        vmax = max(spec[i].max() for i in idxs)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, ch, i in zip(axes.flatten(), show_ch, idxs):
            im = ax.pcolormesh(times, freqs, spec[i], shading="auto",
                               vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(ch)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frequency (Hz)")
            ax.set_ylim(0, 60)
        fig.colorbar(im, ax=axes.ravel(), label="Power (dB)")
        fig.suptitle(f"Day 7.1 example — {c['Segment_ID']} ({c['Class']})",
                     fontsize=14)
        plt.tight_layout()
        png_path = OUT_DIR / "day7_01_spectrogram_example.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    all_ok = raw_ok and prep_ok and spec_ok and cnn_ok and label_ok and leak == 0

    # ---------- save outputs ----------
    if all_ok:
        summary_df = pd.DataFrame(rows_summary)
        summary_df.to_csv(OUT_DIR / "day7_01_test_summary.csv", index=False)
        np.save(OUT_DIR / "day7_01_cnn_input_test.npy", cnn_input)
        label_df.to_csv(OUT_DIR / "day7_01_labels.csv", index=False)

    # ---------- terminal report ----------
    print("\nSampling Rate:")
    print(f"{FS} Hz")
    print("\nSegment Size:")
    print(f"{SEGMENT_SIZE} samples")
    print("\nSegment Duration:")
    print(f"{SEGMENT_SEC} seconds")
    print("\nEEG Channels:")
    print(len(EEG_CHANNELS))
    print("\nSpectrogram:")
    print("65 × 7")

    print("\n" + "-" * 60)
    print("INPUT SHAPES")
    print("-" * 60)
    print("\nRaw EEG:")
    print("(512, 19)")
    print("\nPreprocessed EEG:")
    print("(512, 19)")
    print("\nSpectrogram per channel:")
    print("(65, 7)")
    print("\nCNN Input:")
    print(str(cnn_input.shape))

    print("\n" + "-" * 60)
    print("LABELS")
    print("-" * 60)
    print("\nADHD = 1")
    print("Control = 0")

    print("\n" + "-" * 60)
    print("DATA LEAKAGE")
    print("-" * 60)
    print("\nParticipant Leakage:")
    print(leak)
    if leak == 0:
        print("\nNo participant leakage detected")

    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    print("\nRaw EEG validation:")
    print("PASSED" if raw_ok else "FAILED")
    print("\nPreprocessing validation:")
    print("PASSED" if prep_ok else "FAILED")
    print("\nSpectrogram validation:")
    print("PASSED" if spec_ok else "FAILED")
    print("\nCNN input validation:")
    print("PASSED" if cnn_ok else "FAILED")
    print("\nLabel validation:")
    print("PASSED" if label_ok else "FAILED")
    print("\nLeakage validation:")
    print("PASSED" if leak == 0 else "FAILED")

    if all_ok:
        print("\n" + "-" * 60)
        print("OUTPUT FILES")
        print("-" * 60)
        print(f"\nSaved: {OUT_DIR / 'day7_01_test_summary.csv'}")
        print(f"Saved: {OUT_DIR / 'day7_01_cnn_input_test.npy'} "
              f"shape={cnn_input.shape}")
        print(f"Saved: {OUT_DIR / 'day7_01_labels.csv'}")
        print(f"Saved: {OUT_DIR / 'day7_01_spectrogram_example.png'}")

    print("\n" + "-" * 60)
    if all_ok:
        print("DAY 7.1 STATUS: PASSED")
    else:
        print("DAY 7.1 STATUS: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()