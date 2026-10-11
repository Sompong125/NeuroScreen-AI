# ============================================================
# DAY 6.2 - SEGMENT MANIFEST CREATION
# แบ่ง EEG ของแต่ละ Participant เป็น 4-second segments (512 samples)
# สร้าง Segment Manifest พร้อม validation และ leakage check
#
# ใช้ split files จาก Day 6.1 (ไม่ regenerate split)
# Run: python day6_02_segment_manifest.py
# ============================================================

import sys
from pathlib import Path

import pandas as pd

# ---------- paths ----------
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"

# ---------- settings ----------
FS = 128
SEGMENT_SEC = 4
SEGMENT_SIZE = FS * SEGMENT_SEC  # 512
OVERLAP = 0

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


def natural_key(value):
    import re
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(value)))


def load_splits():
    """โหลด split CSVs จาก Day 6.1"""
    splits = {}
    for name in ["train", "validation", "test"]:
        path = OUT_DIR / f"{name}_participants.csv"
        if not path.exists():
            print(f"ERROR: ไม่พบ {path}")
            print("       กรุณารัน day6_01_participant_split.py ก่อน")
            sys.exit(1)
        df = pd.read_csv(path)
        splits[name.capitalize()] = df
    return splits


def count_participant_samples(dataset_path):
    """นับจำนวน samples ของแต่ละ participant แบบ chunked (ไม่โหลด EEG)"""
    chunks = []
    for chunk in pd.read_csv(dataset_path, usecols=["ID", "Class"], chunksize=500_000):
        counts = chunk.groupby(["ID", "Class"]).size().reset_index(name="Total_Samples")
        chunks.append(counts)

    # รวมผลจากทุก chunk
    combined = pd.concat(chunks, ignore_index=True)
    result = combined.groupby(["ID", "Class"])["Total_Samples"].sum().reset_index()
    return result


def main():
    print("=" * 60)
    print("DAY 6.2 - SEGMENT MANIFEST CREATION")
    print("=" * 60)

    # ---------- load splits ----------
    splits = load_splits()

    # สร้าง mapping ID → Split
    id_to_split = {}
    for split_name, df in splits.items():
        for pid in df["ID"]:
            id_to_split[pid] = split_name

    # ---------- count samples ----------
    path = find_dataset()
    if path is None:
        print("ERROR: ไม่พบ adhdata.csv")
        sys.exit(1)
    print(f"\nDataset: {path}")

    sample_counts = count_participant_samples(path)

    # ---------- create participant segment counts ----------
    sample_counts["Complete_Segments"] = sample_counts["Total_Samples"] // SEGMENT_SIZE
    sample_counts["Remainder_Samples"] = sample_counts["Total_Samples"] % SEGMENT_SIZE
    sample_counts["Split"] = sample_counts["ID"].map(id_to_split)

    # ---------- create segment manifest ----------
    manifest_rows = []
    for _, row in sample_counts.iterrows():
        pid = row["ID"]
        cls = row["Class"]
        split = row["Split"]
        n_segments = row["Complete_Segments"]

        for seg_idx in range(n_segments):
            start = seg_idx * SEGMENT_SIZE
            end = start + SEGMENT_SIZE - 1  # inclusive
            seg_id = f"{pid}_{split}_segment_{seg_idx:04d}"

            manifest_rows.append({
                "Segment_ID": seg_id,
                "Split": split,
                "ID": pid,
                "Class": cls,
                "Segment_Index": seg_idx,
                "Start_Sample": start,
                "End_Sample": end,
                "Num_Samples": SEGMENT_SIZE,
            })

    manifest = pd.DataFrame(manifest_rows)

    # ---------- validation ----------
    print("\n" + "-" * 60)
    print("VALIDATION CHECK")
    print("-" * 60)

    # 1. Segment_ID unique
    dup_seg_ids = manifest["Segment_ID"].duplicated().sum()
    print(f"Duplicate Segment IDs: {dup_seg_ids}")

    # 2. Num_Samples = 512
    invalid_sizes = (manifest["Num_Samples"] != SEGMENT_SIZE).sum()
    print(f"Invalid segment sizes (≠512): {invalid_sizes}")

    # 3. Start_Sample >= 0
    neg_start = (manifest["Start_Sample"] < 0).sum()
    print(f"Negative Start_Sample: {neg_start}")

    # 4. End > Start
    invalid_range = (manifest["End_Sample"] <= manifest["Start_Sample"]).sum()
    print(f"Invalid End ≤ Start: {invalid_range}")

    # 5. Each participant has only 1 split
    p_splits = manifest.groupby("ID")["Split"].nunique()
    multi_split_pids = (p_splits > 1).sum()
    print(f"Participants with multiple splits: {multi_split_pids}")

    # 6. All manifest IDs are in Day 6.1 splits
    all_split_ids = set()
    for df in splits.values():
        all_split_ids.update(df["ID"])
    manifest_ids = set(manifest["ID"])
    unknown_ids = len(manifest_ids - all_split_ids)
    print(f"IDs not in Day 6.1 splits: {unknown_ids}")

    # 7. Leakage check between splits
    train_ids = set(splits["Train"]["ID"])
    val_ids = set(splits["Validation"]["ID"])
    test_ids = set(splits["Test"]["ID"])

    leak_tv = len(train_ids & val_ids)
    leak_tt = len(train_ids & test_ids)
    leak_vt = len(val_ids & test_ids)
    print(f"Train ∩ Validation: {leak_tv}")
    print(f"Train ∩ Test: {leak_tt}")
    print(f"Validation ∩ Test: {leak_vt}")

    # 8. Segment index sequence check
    def check_sequence(group):
        indices = group["Segment_Index"].sort_values().tolist()
        expected = list(range(len(indices)))
        return indices == expected

    valid_sequences = manifest.groupby("ID").apply(check_sequence)
    bad_sequences = int((~valid_sequences).sum())
    print(f"Invalid segment index sequences: {bad_sequences}")

    # 9. Start/End continuity check
    def check_continuity(group):
        if len(group) < 2:
            return True
        group = group.sort_values("Segment_Index")
        for i in range(len(group) - 1):
            curr_end = group.iloc[i]["End_Sample"]
            next_start = group.iloc[i + 1]["Start_Sample"]
            if next_start != curr_end + 1:
                return False
        return True

    valid_continuity = manifest.groupby("ID").apply(check_continuity)
    bad_continuity = int((~valid_continuity).sum())
    print(f"Invalid Start/End continuity: {bad_continuity}")

    # 10. Overall leakage
    participant_leakage = multi_split_pids + leak_tv + leak_tt + leak_vt
    print(f"\nParticipant leakage: {participant_leakage}")
    print(f"Segment ID duplicates: {dup_seg_ids}")
    print(f"Invalid segment sizes: {invalid_sizes}")

    all_ok = (
        dup_seg_ids == 0
        and invalid_sizes == 0
        and neg_start == 0
        and invalid_range == 0
        and multi_split_pids == 0
        and unknown_ids == 0
        and leak_tv == 0
        and leak_tt == 0
        and leak_vt == 0
        and bad_sequences == 0
        and bad_continuity == 0
    )

    if participant_leakage == 0:
        print("\nNo participant leakage detected")
    else:
        print("\nDATA LEAKAGE DETECTED")

    status = "PASSED" if all_ok else "FAILED"

    # ---------- print summary ----------
    print("\n" + "-" * 60)
    print("PARTICIPANT SUMMARY")
    print("-" * 60)
    print("\nTotal Participants:")
    print(len(sample_counts))
    print("\nTrain:")
    print(len(splits["Train"]))
    print("\nValidation:")
    print(len(splits["Validation"]))
    print("\nTest:")
    print(len(splits["Test"]))

    print("\n" + "-" * 60)
    print("SEGMENT SUMMARY")
    print("-" * 60)
    for split_name in ["Train", "Validation", "Test"]:
        n_seg = len(manifest[manifest["Split"] == split_name])
        print(f"\n{split_name}:")
        print(n_seg)
    print("\nTotal:")
    print(len(manifest))

    # ---------- save outputs ----------
    if all_ok:
        # segment_manifest.csv
        manifest_path = OUT_DIR / "segment_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        print(f"\nSaved: {manifest_path}")

        # day6_02_participant_segment_counts.csv
        counts_sorted = sample_counts.sort_values(
            ["Split", "Class", "ID"],
            key=lambda s: s.map(natural_key) if s.name == "ID" else s
        )
        counts_path = OUT_DIR / "day6_02_participant_segment_counts.csv"
        counts_sorted.to_csv(counts_path, index=False)
        print(f"Saved: {counts_path}")

        # day6_02_segment_summary.csv
        summary_rows = []
        for split_name in ["Train", "Validation", "Test"]:
            split_df = sample_counts[sample_counts["Split"] == split_name]
            for cls in ["ADHD", "Control"]:
                cls_df = split_df[split_df["Class"] == cls]
                if len(cls_df) > 0:
                    summary_rows.append({
                        "Split": split_name,
                        "Class": cls,
                        "Participants": len(cls_df),
                        "Total_Segments": cls_df["Complete_Segments"].sum(),
                        "Total_Complete_Samples": cls_df["Complete_Segments"].sum() * SEGMENT_SIZE,
                        "Total_Remainder_Samples": cls_df["Remainder_Samples"].sum(),
                    })
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = OUT_DIR / "day6_02_segment_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Saved: {summary_csv_path}")

        # day6_02_segment_summary.txt
        lines = [
            "=" * 60,
            "DAY 6.2 - SEGMENT MANIFEST SUMMARY",
            "=" * 60,
            f"Sampling Rate: {FS} Hz",
            f"Segment Duration: {SEGMENT_SEC} seconds",
            f"Segment Size: {SEGMENT_SIZE} samples",
            f"Overlap: {OVERLAP} samples",
            f"Total Participants: {len(sample_counts)}",
        ]

        for split_name in ["Train", "Validation", "Test"]:
            split_df = sample_counts[sample_counts["Split"] == split_name]
            n_total = len(split_df)
            n_adhd = len(split_df[split_df["Class"] == "ADHD"])
            n_control = len(split_df[split_df["Class"] == "Control"])
            n_seg = len(manifest[manifest["Split"] == split_name])

            lines += [
                "",
                "-" * 60,
                split_name.upper(),
                "-" * 60,
                f"Participants: {n_total}",
                f"ADHD: {n_adhd}",
                f"Control: {n_control}",
                f"Total Segments: {n_seg}",
            ]

        lines += [
            "",
            "-" * 60,
            "LEAKAGE CHECK",
            "-" * 60,
            f"Participant Leakage: {participant_leakage}",
            f"Segment ID Duplicates: {dup_seg_ids}",
            f"Invalid Segment Size: {invalid_sizes}",
            f"Status: {status}",
            "=" * 60,
        ]

        txt_path = OUT_DIR / "day6_02_segment_summary.txt"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {txt_path}")

    print("\n" + "-" * 60)
    if all_ok:
        print("DAY 6.2 STATUS: PASSED")
    else:
        print("DAY 6.2 STATUS: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()