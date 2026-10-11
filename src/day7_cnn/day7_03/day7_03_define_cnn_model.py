# ============================================================
# DAY 7.3 - CNN MODEL DEFINITION (UNTRAINED) — CLEANUP VERSION
# แก้ไข 2 จุด:
#   FIX #1: รายงาน parameter count จาก weights จริงของ Keras
#           (trainable_weights / non_trainable_weights)
#           พร้อมตรวจ trainable + non_trainable == total
#   FIX #2: forward pass CSV สะอาด (3 columns, 4 rows, ไม่มี
#           comment/HTML/ภาษาไทย/blank row) + มี self-validation
#
# - compile ได้ แต่ห้าม model.fit() / ห้าม train / ห้าม evaluate
# Run: python day7_03_define_cnn_model.py
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv as csv_lib
import sys
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("ERROR: ยังไม่ได้ติดตั้ง TensorFlow")
    print("ติดตั้งด้วยคำสั่ง:  pip install tensorflow")
    sys.exit(1)

# ---------------- CONFIG ----------------
RANDOM_SEED = 42
LEARNING_RATE = 0.001
INPUT_SHAPE = (65, 7, 19)  # (frequency, time, eeg channels)

EXPECTED_SHAPES = {
    "X_train": (2992, 19, 65, 7),
    "X_val": (563, 19, 65, 7),
    "X_test": (618, 19, 65, 7),
}

BASE_DIR = Path(__file__).resolve().parent
NPZ_CANDIDATES = [
    BASE_DIR.parent / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    BASE_DIR.parent.parent / "day7_02" / "day7_02_cnn_dataset.npz",
]

MODEL_PATH = BASE_DIR / "day7_03_cnn_baseline_untrained.keras"
CSV_PATH = BASE_DIR / "day7_03_forward_pass.csv"


def build_model():
    """Baseline CNN ตามสเปก Day 7.3 — ไม่เปลี่ยน architecture ใดๆ"""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="cnn_baseline_untrained")
    return model


def validate_forward_csv(path):
    """FIX #2: ตรวจว่า CSV สะอาดจริง — header ถูก, 4 rows, ไม่มี
    blank/comment/HTML/malformed/NaN/Inf"""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv_lib.reader(f))
    except Exception as exc:  # noqa: BLE001
        return False, f"read error: {exc}"

    if any(len(r) == 0 or all(c.strip() == "" for c in r) for r in rows):
        return False, "blank row found"
    if rows[0] != ["Sample", "True_Label", "Model_Output"]:
        return False, "header mismatch"
    data = rows[1:]
    if len(data) != 4:
        return False, f"expected 4 data rows, got {len(data)}"
    for r in data:
        if len(r) != 3:
            return False, "malformed row"
        for c in r:
            if "<" in c or c.strip().startswith("#"):
                return False, "HTML/comment found"
        try:
            vals = [float(c) for c in r]
        except ValueError:
            return False, "non-numeric value"
        if any(np.isnan(v) or np.isinf(v) for v in vals):
            return False, "NaN/Inf found"
    return True, "ok"


def main():
    print("=" * 60)
    print("DAY 7.3 - CNN MODEL DEFINITION")
    print("=" * 60)

    # ---------- load dataset ----------
    npz_path = next((p for p in NPZ_CANDIDATES if p.exists()), None)
    if npz_path is None:
        print("ERROR: ไม่พบ day7_02_cnn_dataset.npz — รัน Day 7.2 ก่อน")
        sys.exit(1)
    print(f"\nDataset:\n{npz_path.name}")

    data = np.load(npz_path)
    for key, exp in EXPECTED_SHAPES.items():
        if data[key].shape != exp:
            print(f"ERROR: {key}.shape = {data[key].shape} (คาด {exp})")
            sys.exit(1)
    X_train = data["X_train"]
    y_train = data["y_train"]

    Xc_train = X_train.transpose(0, 2, 3, 1)
    Xc_val = data["X_val"].transpose(0, 2, 3, 1)
    Xc_test = data["X_test"].transpose(0, 2, 3, 1)

    print("\n" + "-" * 60)
    print("INPUT")
    print("-" * 60)
    print("\nOriginal shape:")
    print(X_train.shape)
    print("\nCNN input shape:")
    print(Xc_train.shape)
    print("\nSingle sample:")
    print(INPUT_SHAPE)

    # ---------- build + compile ----------
    model = build_model()
    summary_lines = []
    model.summary(print_fn=lambda s: summary_lines.append(s))

    compile_ok = True
    try:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    except Exception as exc:  # noqa: BLE001
        compile_ok = False
        print("ERROR compile:", exc)

    # ---------- FIX #1: parameter count จาก weights จริง ----------
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nontrain_params = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    param_consistent = (trainable_params + nontrain_params == total_params)

    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    print("\nCNN Baseline")
    print("\nInput:")
    print("65 × 7 × 19")
    print("\nConv2D:")
    print("32 filters")
    print("\nConv2D:")
    print("64 filters")
    print("\nConv2D:")
    print("128 filters")
    print("\nGlobalAveragePooling2D")
    print("\nDense:")
    print("64")
    print("\nOutput:")
    print("1 sigmoid")

    print("\n" + "-" * 60)
    print("PARAMETERS")
    print("-" * 60)
    print("\nTotal parameters:")
    print(total_params)
    print("\nTrainable parameters:")
    print(trainable_params)
    print("\nNon-trainable parameters:")
    print(nontrain_params)
    print("\nParameter consistency:")
    print(f"{trainable_params} + {nontrain_params} = {total_params}")
    print("PASSED" if param_consistent else "FAILED")

    # ---------- forward pass ----------
    fp_ok = True
    range_ok = False
    out_shape_ok = False
    pred = None
    try:
        sample = np.ascontiguousarray(X_train[:1].transpose(0, 2, 3, 1))
        pred = model(sample, training=False).numpy()
        out_shape_ok = pred.shape == (1, 1)
        range_ok = bool(np.all((pred >= 0.0) & (pred <= 1.0)))
    except Exception as exc:  # noqa: BLE001
        fp_ok = False
        print("ERROR forward pass:", exc)

    sample4 = np.ascontiguousarray(X_train[:4].transpose(0, 2, 3, 1))
    preds4 = model(sample4, training=False).numpy().flatten()

    print("\n" + "-" * 60)
    print("FORWARD PASS")
    print("-" * 60)
    print("\nInput:")
    print(sample.shape)
    print("\nOutput:")
    print(pred.shape if pred is not None else "ERROR")
    print("\nOutput range:")
    print("0 <= output <= 1" if range_ok else "OUT OF RANGE")

    # ---------- core tests ----------
    input_ok = (model.input_shape == (None,) + INPUT_SHAPE
                and Xc_train.shape == (2992, 65, 7, 19)
                and Xc_val.shape == (563, 65, 7, 19)
                and Xc_test.shape == (618, 65, 7, 19))
    core_tests = [
        ("Input shape validation", input_ok),
        ("Output shape validation", out_shape_ok),
        ("Model creation", True),
        ("Model compile", compile_ok),
        ("Forward pass", fp_ok),
        ("Output range", range_ok),
        ("Parameter consistency", param_consistent),
    ]
    core_ok = all(ok for _, ok in core_tests)

    # ---------- save + reload + write files ----------
    save_ok = False
    reload_ok = False
    csv_ok = False
    csv_reason = "not executed"
    if core_ok:
        try:
            model.save(MODEL_PATH)
            save_ok = True
            loaded = keras.models.load_model(MODEL_PATH)
            pred_r = loaded(sample, training=False).numpy()
            reload_ok = (pred_r.shape == (1, 1)
                         and bool(np.allclose(pred, pred_r, atol=1e-6)))
        except Exception as exc:  # noqa: BLE001
            print("ERROR save/reload:", exc)

        # model summary txt
        lines = [
            "=" * 60,
            "DAY 7.3 - MODEL SUMMARY (UNTRAINED)",
            "=" * 60,
            f"Input shape: {INPUT_SHAPE}  (frequency=65, time=7, eeg channels=19)",
            "Output shape: (1,) sigmoid  [0=Control, 1=ADHD]",
            f"Total parameters: {total_params}",
            f"Trainable parameters: {trainable_params}",
            f"Non-trainable parameters: {nontrain_params}",
            f"Parameter consistency: {trainable_params} + {nontrain_params} "
            f"= {total_params} -> {'PASSED' if param_consistent else 'FAILED'}",
            "Optimizer: Adam(lr=0.001) | Loss: binary_crossentropy | Metrics: accuracy",
            "Model status: UNTRAINED (ยังไม่ได้เรียก model.fit())",
            "",
        ] + summary_lines + ["=" * 60]
        (BASE_DIR / "day7_03_model_summary.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

        # FIX #2: forward pass CSV สะอาด — ข้อมูลล้วน ไม่มี note ใดๆ
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv_lib.writer(f)
            w.writerow(["Sample", "True_Label", "Model_Output"])
            for i in range(4):
                w.writerow([i, int(y_train[i]), round(float(preds4[i]), 6)])

        csv_ok, csv_reason = validate_forward_csv(CSV_PATH)

        # validation summary txt (note เรื่อง UNTRAINED อยู่ที่นี่เท่านั้น)
        vlines = ["=" * 60, "DAY 7.3 - VALIDATION SUMMARY", "=" * 60]
        for name, ok in core_tests:
            vlines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
        vlines.append(f"[{'PASS' if save_ok else 'FAIL'}] Save model")
        vlines.append(f"[{'PASS' if reload_ok else 'FAIL'}] Reload model")
        vlines.append(f"[{'PASS' if csv_ok else 'FAIL'}] Forward pass CSV "
                      f"validation ({csv_reason})")
        vlines += [
            "",
            f"Forward pass outputs (4 samples): "
            + ", ".join(str(round(float(v), 6)) for v in preds4),
            "NOTE: Model is UNTRAINED (Day 7.3).",
            "Model_Output has no clinical/diagnostic meaning yet.",
            "หมายเหตุ: Model ยังไม่ได้ train — ค่า output เป็นผลจากน้ำหนักสุ่มเท่านั้น",
            "Class imbalance (Train: ADHD 1687 / Control 1305):",
            "  Class imbalance will be considered during the training phase.",
            "Training performed: NO",
            "Model status: UNTRAINED",
            f"Saved model: {MODEL_PATH.name}",
            "=" * 60,
        ]
        (BASE_DIR / "day7_03_validation_summary.txt").write_text(
            "\n".join(vlines) + "\n", encoding="utf-8")

    tests = core_tests + [
        ("Save model", save_ok),
        ("Reload model", reload_ok),
        ("Forward pass CSV validation", csv_ok),
    ]
    all_ok = all(ok for _, ok in tests)

    # ---------- terminal report ----------
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    print("\nTraining performed:")
    print("NO")
    print("\nModel status:")
    print("UNTRAINED")
    print("\nClass imbalance will be considered during the training phase.")

    print("\n" + "-" * 60)
    print("STATUS")
    print("-" * 60)
    if all_ok:
        print(f"\nSaved: {BASE_DIR / 'day7_03_model_summary.txt'}")
        print(f"Saved: {CSV_PATH}")
        print(f"Saved: {BASE_DIR / 'day7_03_validation_summary.txt'}")
        print(f"Saved: {MODEL_PATH}")
        print("\nDAY 7.3: PASSED")
    else:
        print("\nDAY 7.3: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()