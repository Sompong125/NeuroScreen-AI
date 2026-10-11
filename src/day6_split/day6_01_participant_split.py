# ============================================================
# DAY 6.1 - PARTICIPANT LEVEL SPLIT
# แบ่ง Participant IDs เป็น Train / Validation / Test (70/15/15)
# แบบ stratified, seed=42, ห้ามมี ID ซ้ำข้าม split (กัน data leakage)
#
# อ่านเฉพาะคอลัมน์ ID, Class แบบ chunked (ไม่โหลด EEG ทั้งไฟล์)
# Run: python day6_01_participant_split.py
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
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EXPECTED_TOTAL = 121
EXPECTED_ADHD = 61
EXPECTED_CONTROL = 60
EXPECTED = {
    "Train":      {"Total": 85, "ADHD": 43, "Control": 42},
    "Validation": {"Total": 18, "ADHD": 9,  "Control": 9},
    "Test":       {"Total": 18, "ADHD": 9,  "Control": 9},
}

CANDIDATES = [
    BASE_DIR.parent / "data" / "adhdata.csv",
    BASE_DIR.parent / "adhdata.csv",
    BASE_DIR.parent.parent / "data" / "adhdata.csv",
    BASE_DIR / "data" / "adhdata.csv",
]


def natural_key(value):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(value))]


def find_dataset():
    for p in CANDIDATES:
        if p.exists():
            return p
    return None


def load_participants(path):
    """อ่านเฉพาะ ID, Class แบบ chunked → ตาราง participant-level"""
    chunks = []
    for chunk in pd.read_csv(path, usecols=["ID", "Class"], chunksize=500_000):
        chunks.append(chunk.drop_duplicates())
    pairs = pd.concat(chunks, ignore_index=True)

    # ตรวจว่าแต่ละ participant มี Class เดียวจริง
    n_class_per_id = pairs.groupby("ID")["Class"].nunique()
    bad_ids = n_class_per_id[n_class_per_id > 1]

    participants = (
        pairs.drop_duplicates(subset=["ID"])
        .sort_values("ID", key=lambda s: s.map(natural_key))
        .reset_index(drop=True)
    )
    return participants, bad_ids


def stratified_split(participants, seed=SEED):
    """แบ่งราย class เพื่อรักษาสัดส่วน ADHD/Control (stratified)"""
    rng = np.random.default_rng(seed)
    splits = {"Train": [], "Validation": [], "Test": []}

    for cls in sorted(participants["Class"].unique()):
        ids = np.array(
            sorted(participants.loc[participants["Class"] == cls, "ID"].tolist(),
                   key=natural_key)
        )
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(TRAIN_RATIO * n))
        n_val = int(round(VAL_RATIO * n))
        splits["Train"].extend(ids[:n_train].tolist())
        splits["Validation"].extend(ids[n_train:n_train + n_val].tolist())
        splits["Test"].extend(ids[n_train + n_val:].tolist())
    return splits


def split_df(participants, ids):
    df = participants[participants["ID"].isin(set(ids))][["ID", "Class"]].copy()
    return df.sort_values("ID", key=lambda s: s.map(natural_key)).reset_index(drop=True)


def count_class(df):
    vc = df["Class"].value_counts()
    return int(vc.get("ADHD", 0)), int(vc.get("Control", 0))


def main():
    print("=" * 60)
    print("DAY 6.1 - PARTICIPANT LEVEL SPLIT")
    print("=" * 60)

    path = find_dataset()
    if path is None:
        print("ERROR: ไม่พบ adhdata.csv ได้ลองค้นหาที่:")
        for p in CANDIDATES:
            print("  ", p)
        sys.exit(1)
    print(f"\nDataset path: {path}")

    participants, bad_ids = load_participants(path)
    adhd, control = count_class(participants)

    print("\nTotal participants:")
    print(len(participants))
    print("\nADHD:")
    print(adhd)
    print("\nControl:")
    print(control)

    splits = stratified_split(participants)
    dfs = {k: split_df(participants, v) for k, v in splits.items()}

    for name in ["Train", "Validation", "Test"]:
        a, c = count_class(dfs[name])
        print("\n" + "-" * 60)
        print(name.upper())
        print("-" * 60)
        print("\nTotal:")
        print(len(dfs[name]))
        print("\nADHD:")
        print(a)
        print("\nControl:")
        print(c)

    # ---------- leakage check ----------
    s_train, s_val, s_test = (set(splits[k]) for k in ["Train", "Validation", "Test"])
    inter_tv = len(s_train & s_val)
    inter_tt = len(s_train & s_test)
    inter_vt = len(s_val & s_test)
    union_ids = s_train | s_val | s_test

    print("\n" + "-" * 60)
    print("LEAKAGE CHECK")
    print("-" * 60)
    print("\nTrain ∩ Validation:")
    print(inter_tv)
    print("\nTrain ∩ Test:")
    print(inter_tt)
    print("\nValidation ∩ Test:")
    print(inter_vt)
    print("\nTotal unique IDs:")
    print(len(union_ids))
    print("-" * 60)

    # ---------- validation checks ----------
    a_tr, c_tr = count_class(dfs["Train"])
    a_va, c_va = count_class(dfs["Validation"])
    a_te, c_te = count_class(dfs["Test"])

    checks = [
        ("Total participants = 121", len(participants) == EXPECTED_TOTAL),
        ("ADHD = 61", adhd == EXPECTED_ADHD),
        ("Control = 60", control == EXPECTED_CONTROL),
        ("Train = 85", len(dfs["Train"]) == EXPECTED["Train"]["Total"]),
        ("Validation = 18", len(dfs["Validation"]) == EXPECTED["Validation"]["Total"]),
        ("Test = 18", len(dfs["Test"]) == EXPECTED["Test"]["Total"]),
        ("Train + Validation + Test = 121",
         len(dfs["Train"]) + len(dfs["Validation"]) + len(dfs["Test"]) == EXPECTED_TOTAL),
        ("ไม่มี ID ซ้ำระหว่าง splits",
         inter_tv == 0 and inter_tt == 0 and inter_vt == 0),
        ("ทุก participant มี Class เดียว", len(bad_ids) == 0),
        ("Union ของ splits ครบ 121 คน", union_ids == set(participants["ID"])),
        ("Train class dist = 43/42", (a_tr, c_tr) == (43, 42)),
        ("Validation class dist = 9/9", (a_va, c_va) == (9, 9)),
        ("Test class dist = 9/9", (a_te, c_te) == (9, 9)),
    ]

    print("\nVALIDATION CHECKS")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'✓' if ok else '✗'}] {name}")
        all_ok = all_ok and ok

    status = "PASSED" if all_ok else "FAILED"
    print("\nStatus:")
    print(status)
    print("=" * 60)

    # ---------- save outputs ----------
    if all_ok:
        for fname, key in [
            ("train_participants.csv", "Train"),
            ("validation_participants.csv", "Validation"),
            ("test_participants.csv", "Test"),
        ]:
            dfs[key].to_csv(OUT_DIR / fname, index=False)
            print(f"Saved: {OUT_DIR / fname}")

        summary = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Total": [len(dfs[k]) for k in ["Train", "Validation", "Test"]],
            "ADHD": [count_class(dfs[k])[0] for k in ["Train", "Validation", "Test"]],
            "Control": [count_class(dfs[k])[1] for k in ["Train", "Validation", "Test"]],
        })
        summary.to_csv(OUT_DIR / "day6_split_summary.csv", index=False)
        print(f"Saved: {OUT_DIR / 'day6_split_summary.csv'}")

        lines = [
            "=" * 60,
            "DAY 6.1 - PARTICIPANT LEVEL SPLIT SUMMARY",
            "=" * 60,
            f"Dataset path: {path}",
            f"Split ratio: 70/15/15 | Random seed: {SEED}",
            f"Total participants: {len(participants)} (ADHD {adhd} / Control {control})",
            "",
        ]
        for key in ["Train", "Validation", "Test"]:
            a, c = count_class(dfs[key])
            lines += [
                f"[{key}] Total={len(dfs[key])} ADHD={a} Control={c}",
                "  IDs: " + ", ".join(dfs[key]["ID"].tolist()),
                "",
            ]
        lines += [
            f"Leakage check: Train∩Val={inter_tv}, Train∩Test={inter_tt}, "
            f"Val∩Test={inter_vt}, Union={len(union_ids)}",
            f"Status: {status}",
            "=" * 60,
        ]
        txt_path = OUT_DIR / "day6_01_split_summary.txt"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {txt_path}")

    if not all_ok:
        print("\nDAY 6.1 STATUS: FAILED")
        sys.exit(1)

    print("\nDAY 6.1 STATUS: PASSED")


if __name__ == "__main__":
    main()