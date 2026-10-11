# ============================================================
# DAY 7.2 - CNN DATASET PREPARATION (FULL DATASET)
# สร้าง CNN dataset จริงจากทุก segment ใน segment_manifest.csv
#
# Raw EEG → 50 Hz Notch → Per-segment Per-channel Standardization
#           → Spectrogram (19, 65, 7) → stack เป็น X_train/val/test
#
# - อ่าน CSV แบบ chunked (single pass) ไม่โหลดทั้ง dataset เข้า RAM
# - Start/End_Sample เป็น participant-local index (ไม่ใช่ global row)
# - ยังไม่สร้าง CNN / ไม่ train / ไม่ evaluate
# Run: python day7_02_prepare_cnn_dataset.py
# ============================================================

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, iirnotch, spectrogram

# ---------------- CONFIG (reproducibility) ----------------
RANDOM_SEED = 42
FS = 128
SEGMENT_SIZE = 512
N_CHANNELS = 19
N_FREQ_BINS = 65
N_TIME_BINS = 7
NOTCH_FREQ = 50
NOTCH_Q = 30
NPERSEG = 128
NOVERLAP = 64
CHUNK_SIZE = 100_000

EXPECTED_SEGMENTS = {"Train": 2992, "Validation": 563, "Test": 618}
EXPECTED_PARTICIPANTS = {"Train": 85, "Validation": 18, "Test": 18}
EXPECTED_CLASS_PARTS = {
    "Train": {"ADHD": 43, "Control": 42},
    "Validation": {"ADHD": 9, "Control": 9},
    "Test": {"ADHD": 9, "Control": 9},
}
CLASS_TO_LABEL = {"ADHD": 1, "Control": 0}

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

BASE_DIR = Path(__file__).resolve().parent

DATASET_CANDIDATES = [
    BASE_DIR.parent.parent / "adhdata.csv",
    BASE_DIR.parent.parent / "data" / "adhdata.csv",
    BASE_DIR.parent.parent.parent / "data" / "adhdata.csv",
    BASE_DIR / "data" / "adhdata.csv",
]
MANIFEST_CANDIDATES = [
    BASE_DIR.parent.parent / "day6_split" / "output",
    BASE_DIR.parent.parent.parent / "day6_split" / "output",
    BASE_DIR.parent / "day6_split" / "output",
]

NOTCH_B, NOTCH_A = iirnotch(w0=NOTCH_FREQ, Q=NOTCH_Q, fs=FS)


def natural_key(value):
    import re
    return tuple(int(t) if t.isdigit() else t
                 for t in re.split(r"(\d+)", str(value)))


def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load_targets(manifest_path):
    """สร้าง target segments ราย participant จาก manifest (Day 6.2)"""
    manifest = pd.read_csv(manifest_path)
    targets = {}
    for _, row in manifest.iterrows():
        pid = row["ID"]
        t = targets.setdefault(pid, {
            "split": row["Split"], "cls": row["Class"],
            "n": 0, "seg_ids": [],
        })
        t["n"] += 1
        t["seg_ids"].append((row["Segment_Index"], row["Segment_ID"]))
    # ตรวจความต่อเนื่องของ segment index (0..n-1)
    for pid, t in targets.items():
        idxs = sorted(i for i, _ in t["seg_ids"])
        if idxs != list(range(t["n"])):
            raise ValueError(f"Manifest ของ {pid} มี segment index ไม่ต่อเนื่อง: {idxs}")
        t["seg_ids"] = [sid for _, sid in sorted(t["seg_ids"])]
    return manifest, targets


def process_segment(seg):
    """seg (512,19) → notch → standardize → spectrogram → (19,65,7) float32"""
    if not np.isfinite(seg).all():
        raise ValueError("พบ NaN/Inf ใน raw segment")
    filtered = np.empty_like(seg)
    for i in range(seg.shape[1]):
        filtered[:, i] = filtfilt(NOTCH_B, NOTCH_A, seg[:, i])
    mean = filtered.mean(axis=0)
    std = filtered.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # ป้องกัน std = 0
    processed = (filtered - mean) / std
    specs = []
    for i in range(processed.shape[1]):
        f, t, S = spectrogram(
            processed[:, i], fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
            scaling="density", mode="psd")
        specs.append(10.0 * np.log10(S + 1e-12))
    tensor = np.stack(specs, axis=0)
    if tensor.shape != (N_CHANNELS, N_FREQ_BINS, N_TIME_BINS):
        raise ValueError(f"Spectrogram shape ผิด: {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("พบ NaN/Inf ใน spectrogram")
    return tensor.astype(np.float32)


def stream_and_build(dataset_path, targets):
    """single-pass chunked streaming + buffer ราย participant"""
    results = {"Train": [], "Validation": [], "Test": []}
    buffers = {pid: [] for pid in targets}
    buf_len = {pid: 0 for pid in targets}
    done_cnt = {pid: 0 for pid in targets}
    examples = {"ADHD": None, "Control": None}
    total_target = sum(t["n"] for t in targets.values())
    total_done = 0

    for chunk in pd.read_csv(dataset_path, usecols=["ID"] + EEG_CHANNELS,
                             chunksize=CHUNK_SIZE):
        for pid, grp in chunk.groupby("ID", sort=False):
            if pid not in targets:
                continue
            t = targets[pid]
            if done_cnt[pid] >= t["n"]:
                continue
            arr = grp[EEG_CHANNELS].to_numpy(dtype=np.float64)
            buffers[pid].append(arr)
            buf_len[pid] += arr.shape[0]
            while done_cnt[pid] < t["n"] and buf_len[pid] >= SEGMENT_SIZE:
                buf = (buffers[pid][0] if len(buffers[pid]) == 1
                       else np.concatenate(buffers[pid]))
                seg = buf[:SEGMENT_SIZE]
                rest = buf[SEGMENT_SIZE:]
                buffers[pid] = [rest] if rest.shape[0] else []
                buf_len[pid] = rest.shape[0]
                tensor = process_segment(seg)
                seg_id = t["seg_ids"][done_cnt[pid]]
                results[t["split"]].append({
                    "segment_id": seg_id, "pid": pid, "cls": t["cls"],
                    "tensor": tensor,
                })
                if examples[t["cls"]] is None:
                    examples[t["cls"]] = (seg_id, tensor)
                done_cnt[pid] += 1
                total_done += 1
                if total_done % 500 == 0:
                    a = len(results["Train"]); b = len(results["Validation"])
                    c = len(results["Test"])
                    print(f"[progress] {total_done}/{total_target} "
                          f"(Train {a}/{EXPECTED_SEGMENTS['Train']}, "
                          f"Val {b}/{EXPECTED_SEGMENTS['Validation']}, "
                          f"Test {c}/{EXPECTED_SEGMENTS['Test']})")
        if total_done == total_target:
            break

    missing = [pid for pid in targets if done_cnt[pid] < targets[pid]["n"]]
    return results, examples, missing


def main():
    print("=" * 60)
    print("DAY 7.2 - CNN DATASET PREPARATION")
    print("=" * 60)

    dataset_path = find_first_existing(DATASET_CANDIDATES)
    manifest_dir = find_first_existing(MANIFEST_CANDIDATES)
    if dataset_path is None or manifest_dir is None:
        print("ERROR: ไม่พบ adhdata.csv หรือ day6_split/output/")
        sys.exit(1)
    manifest_path = manifest_dir / "segment_manifest.csv"
    print(f"\nDataset:\n{dataset_path}")
    print(f"\nManifest:\n{manifest_path}")

    manifest, targets = load_targets(manifest_path)
    split_counts = manifest["Split"].value_counts()
    for split, exp in EXPECTED_SEGMENTS.items():
        got = int(split_counts.get(split, 0))
        if got != exp:
            print(f"ERROR: manifest มี {split} = {got} segments (คาด {exp}) — หยุด")
            sys.exit(1)

    print("\nProcessing dataset (single pass, chunked) ...")
    results, examples, missing = stream_and_build(dataset_path, targets)
    if missing:
        print("ERROR: ได้ segments ไม่ครบสำหรับ participants:", missing)
        sys.exit(1)

    for split in ["Train", "Validation", "Test"]:
        print(f"{split} completed: {len(results[split])} segments")

    # ---------- stack เป็น arrays ----------
    data = {}
    for split, key in [("Train", "train"), ("Validation", "val"), ("Test", "test")]:
        recs = sorted(results[split],
                      key=lambda r: (natural_key(r["pid"]), r["segment_id"]))
        data[f"X_{key}"] = np.stack([r["tensor"] for r in recs], axis=0)
        data[f"y_{key}"] = np.array([CLASS_TO_LABEL[r["cls"]] for r in recs],
                                    dtype=np.int64)
        data[f"ids_{key}"] = np.array([r["segment_id"] for r in recs])
        data[f"pids_{key}"] = [r["pid"] for r in recs]

    # ---------- VALIDATION TESTS 1-10 ----------
    tests = []
    ok_shape = all(
        data[f"X_{k}"].shape[1:] == (N_CHANNELS, N_FREQ_BINS, N_TIME_BINS)
        for k in ["train", "val", "test"])
    tests.append(("Shape validation", ok_shape))

    ok_count = (len(data["X_train"]) == EXPECTED_SEGMENTS["Train"]
                and len(data["X_val"]) == EXPECTED_SEGMENTS["Validation"]
                and len(data["X_test"]) == EXPECTED_SEGMENTS["Test"])
    tests.append(("Segment count validation", ok_count))

    ok_label = all(set(np.unique(data[f"y_{k}"])) <= {0, 1}
                   for k in ["train", "val", "test"])
    tests.append(("Label validation", ok_label))

    ok_nan = all(not np.isnan(data[f"X_{k}"]).any() for k in ["train", "val", "test"])
    tests.append(("NaN validation", ok_nan))

    ok_inf = all(not np.isinf(data[f"X_{k}"]).any() for k in ["train", "val", "test"])
    tests.append(("Inf validation", ok_inf))

    pid_sets = {k: set(data[f"pids_{k}"]) for k in ["train", "val", "test"]}
    leak = (len(pid_sets["train"] & pid_sets["val"])
            + len(pid_sets["train"] & pid_sets["test"])
            + len(pid_sets["val"] & pid_sets["test"]))
    tests.append(("Participant leakage", leak == 0))

    ok_pcount = (len(pid_sets["train"]) == EXPECTED_PARTICIPANTS["Train"]
                 and len(pid_sets["val"]) == EXPECTED_PARTICIPANTS["Validation"]
                 and len(pid_sets["test"]) == EXPECTED_PARTICIPANTS["Test"])
    tests.append(("Participant count", ok_pcount))

    ok_class = True
    for k, split in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        sub = manifest[manifest["Split"] == split]
        for cls in ["ADHD", "Control"]:
            got = len({p for p in data[f"pids_{k}"]
                       if CLASS_TO_LABEL.get(
                           sub.loc[sub['ID'] == p, 'Class'].iloc[0], -1)
                       == CLASS_TO_LABEL[cls]})
            got = len(sub[sub["Class"] == cls]["ID"].unique())
            if got != EXPECTED_CLASS_PARTS[split][cls]:
                ok_class = False
    tests.append(("Class consistency", ok_class))

    ok_seglabel = True
    pid_label = {}
    for k in ["train", "val", "test"]:
        for rec_pid, y in zip(data[f"pids_{k}"], data[f"y_{k}"]):
            if pid_label.setdefault(rec_pid, y) != y:
                ok_seglabel = False
    tests.append(("Segment label consistency", ok_seglabel))

    ok_finite = all(np.isfinite(data[f"X_{k}"]).all() for k in ["train", "val", "test"])
    tests.append(("Value sanity (finite)", ok_finite))

    # ---------- SAVE ----------
    all_ok = all(ok for _, ok in tests)
    npz_path = BASE_DIR / "day7_02_cnn_dataset.npz"
    reload_ok = False
    if all_ok:
        np.savez_compressed(
            npz_path,
            X_train=data["X_train"], y_train=data["y_train"],
            X_val=data["X_val"], y_val=data["y_val"],
            X_test=data["X_test"], y_test=data["y_test"],
            ids_train=data["ids_train"], ids_val=data["ids_val"],
            ids_test=data["ids_test"],
        )
        chk = np.load(npz_path)
        reload_ok = (chk["X_train"].shape == data["X_train"].shape
                     and chk["X_val"].shape == data["X_val"].shape
                     and chk["X_test"].shape == data["X_test"].shape
                     and (chk["y_train"] == data["y_train"]).all()
                     and (chk["y_val"] == data["y_val"]).all()
                     and (chk["y_test"] == data["y_test"]).all())
        tests.append(("Reload validation", reload_ok))
        all_ok = all_ok and reload_ok

        # ---- dataset summary csv ----
        rows = []
        for k, split in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
            sub = manifest[manifest["Split"] == split]
            rows.append({
                "Split": split,
                "Participants": len(pid_sets[k]),
                "ADHD Participants": int((sub.groupby("ID")["Class"].first() == "ADHD").sum()),
                "Control Participants": int((sub.groupby("ID")["Class"].first() == "Control").sum()),
                "Segments": len(data[f"X_{k}"]),
                "ADHD Segments": int((data[f"y_{k}"] == 1).sum()),
                "Control Segments": int((data[f"y_{k}"] == 0).sum()),
                "X Shape": str(data[f"X_{k}"].shape),
                "Label Shape": str(data[f"y_{k}"].shape),
            })
        pd.DataFrame(rows).to_csv(BASE_DIR / "day7_02_dataset_summary.csv", index=False)

        # ---- participant summary csv ----
        prow = []
        for pid, t in targets.items():
            prow.append({"ID": pid, "Class": t["cls"], "Split": t["split"],
                         "Number of Segments": t["n"]})
        pdf = pd.DataFrame(prow)
        pdf = pdf.sort_values(["Split", "Class", "ID"],
                              key=lambda s: s.map(natural_key) if s.name == "ID" else s)
        pdf.to_csv(BASE_DIR / "day7_02_participant_summary.csv", index=False)

        # ---- sample spectrogram png (ADHD + Control) ----
        if examples["ADHD"] and examples["Control"]:
            show_ch = ["Fp1", "Fp2", "F3", "F4"]
            idxs = [EEG_CHANNELS.index(c) for c in show_ch]
            tensors = [examples["ADHD"], examples["Control"]]
            vmin = min(t[1][i].min() for t in tensors for i in idxs)
            vmax = max(t[1][i].max() for t in tensors for i in idxs)
            f_ref = np.arange(N_FREQ_BINS) * (FS / NPERSEG)
            fig, axes = plt.subplots(2, 4, figsize=(16, 7))
            for r, (seg_id, tensor) in enumerate(tensors):
                for c, (ch, i) in enumerate(zip(show_ch, idxs)):
                    ax = axes[r][c]
                    im = ax.pcolormesh(np.arange(N_TIME_BINS), f_ref, tensor[i],
                                       shading="auto", vmin=vmin, vmax=vmax,
                                       cmap="viridis")
                    ax.set_title(f"{ch}")
                    ax.set_ylim(0, 60)
                    ax.set_xlabel("Time bin")
                    ax.set_ylabel("Frequency (Hz)")
                axes[r][0].set_ylabel(f"{tensors[r][0].split('_')[0]}\n({tensors[r][0]})\nFreq (Hz)")
            fig.colorbar(im, ax=axes.ravel(), label="Power (dB)")
            fig.suptitle("Day 7.2 samples — top: ADHD, bottom: Control", fontsize=14)
            plt.tight_layout()
            plt.savefig(BASE_DIR / "day7_02_sample_spectrogram.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---- validation summary txt ----
        lines = ["=" * 60, "DAY 7.2 - VALIDATION SUMMARY", "=" * 60]
        for k, split in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
            X = data[f"X_{k}"]
            lines += [
                f"[{split}] segments={len(X)} shape={X.shape}",
                f"  min={X.min():.4f} max={X.max():.4f} "
                f"mean={X.mean():.4f} std={X.std():.4f}",
                f"  ADHD segments={int((data[f'y_{k}'] == 1).sum())} "
                f"Control segments={int((data[f'y_{k}'] == 0).sum())}",
            ]
        lines += ["", "Tests:"]
        for name, ok in tests:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        lines += ["", f"Participant leakage intersections: {leak}",
                  f"Status: {'PASSED' if all_ok else 'FAILED'}", "=" * 60]
        (BASE_DIR / "day7_02_validation_summary.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    # ---------- TERMINAL REPORT ----------
    print("\n" + "-" * 60)
    print("DATASET")
    print("-" * 60)
    for k, split in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        print(f"\n{split}:")
        print(f"{len(data[f'X_{k}'])} segments")
        print(f"{len(pid_sets[k])} participants")
    print(f"\nTotal:")
    print(f"{len(data['X_train']) + len(data['X_val']) + len(data['X_test'])} segments")
    print(f"{len(pid_sets['train'] | pid_sets['val'] | pid_sets['test'])} participants")

    print("\n" + "-" * 60)
    print("SHAPES")
    print("-" * 60)
    print("\nX_train:")
    print(data["X_train"].shape)
    print("\nX_val:")
    print(data["X_val"].shape)
    print("\nX_test:")
    print(data["X_test"].shape)

    print("\n" + "-" * 60)
    print("LABELS")
    print("-" * 60)
    for k, split in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        y = data[f"y_{k}"]
        print(f"\n{split}:")
        print(f"ADHD = {int((y == 1).sum())}")
        print(f"Control = {int((y == 0).sum())}")

    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "-" * 60)
    print("STATUS")
    print("-" * 60)
    if all_ok:
        print(f"\nSaved: {npz_path}")
        print(f"Saved: {BASE_DIR / 'day7_02_dataset_summary.csv'}")
        print(f"Saved: {BASE_DIR / 'day7_02_participant_summary.csv'}")
        print(f"Saved: {BASE_DIR / 'day7_02_validation_summary.txt'}")
        print(f"Saved: {BASE_DIR / 'day7_02_sample_spectrogram.png'}")
        print("\nDAY 7.2: PASSED")
    else:
        print("\nDAY 7.2: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()