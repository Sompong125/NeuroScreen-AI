# ============================================================
# DAY 7.4 - CNN TRAINING
# Train CNN Baseline (จาก Day 7.3) ด้วย dataset จาก Day 7.2
#
# - ใช้ split เดิมทั้งหมด (participant-level) ไม่สุ่มใหม่
# - class_weight คำนวณจาก TRAIN labels เท่านั้น
# - EarlyStopping + ModelCheckpoint (monitor val_loss)
# - ไม่ใช้ Test set ระหว่าง training / ไม่ใช้เลือก model
# - ยังไม่ทำ final evaluation (Day 7.5)
# Run: python day7_04_train_cnn.py
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv as csv_lib
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("ERROR: ยังไม่ได้ติดตั้ง TensorFlow ใน environment นี้")
    sys.exit(1)

# ---------------- seeds ----------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------- config ----------------
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 5
INPUT_SHAPE = (65, 7, 19)

BASE_DIR = Path(__file__).resolve().parent

NPZ_CANDIDATES = [
    BASE_DIR.parent / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    BASE_DIR.parent.parent / "day7_02" / "day7_02_cnn_dataset.npz",
]
UNTRAINED_CANDIDATES = [
    BASE_DIR.parent / "day7_03" / "day7_03_cnn_baseline_untrained.keras",
    BASE_DIR / "day7_03_cnn_baseline_untrained.keras",
    BASE_DIR.parent.parent / "day7_03" / "day7_03_cnn_baseline_untrained.keras",
]
SPLIT_DIR_CANDIDATES = [
    BASE_DIR.parent.parent / "day6_split" / "output",
    BASE_DIR.parent.parent.parent / "day6_split" / "output",
]

BEST_MODEL_PATH = BASE_DIR / "day7_04_best_model.keras"
HISTORY_CSV = BASE_DIR / "day7_04_training_history.csv"
CLASS_WEIGHT_CSV = BASE_DIR / "day7_04_class_weights.csv"
TRAIN_SUMMARY = BASE_DIR / "day7_04_training_summary.txt"
VALID_SUMMARY = BASE_DIR / "day7_04_validation_summary.txt"
CURVE_PNG = BASE_DIR / "day7_04_training_curve.png"


def find_first(cands):
    return next((p for p in cands if p.exists()), None)


def pids_from_seg_ids(ids):
    return {str(s).split("_")[0] for s in ids}


def main():
    print("=" * 60)
    print("DAY 7.4 - CNN TRAINING")
    print("=" * 60)

    # ---------- load dataset ----------
    npz_path = find_first(NPZ_CANDIDATES)
    if npz_path is None:
        print("ERROR: ไม่พบ day7_02_cnn_dataset.npz")
        sys.exit(1)
    print(f"\nDataset:\n{npz_path.name}")

    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    ds_ok = (X_train.shape == (2992, 19, 65, 7)
             and X_val.shape == (563, 19, 65, 7)
             and X_test.shape == (618, 19, 65, 7))
    tr_ok = X_train.shape == (2992, 19, 65, 7)
    va_ok = X_val.shape == (563, 19, 65, 7)
    te_ok = X_test.shape == (618, 19, 65, 7)

    Xc_train = X_train.transpose(0, 2, 3, 1)
    Xc_val = X_val.transpose(0, 2, 3, 1)
    # X_test transpose สำหรับตรวจ shape เท่านั้น ไม่ใช้ระหว่าง training
    Xc_test = X_test.transpose(0, 2, 3, 1)

    label_ok = (set(np.unique(y_train)) <= {0, 1}
                and set(np.unique(y_val)) <= {0, 1}
                and set(np.unique(y_test)) <= {0, 1})

    n_adhd = int((y_train == 1).sum())
    n_ctrl = int((y_train == 0).sum())

    # ---------- participant split preserved ----------
    pid_train = pids_from_seg_ids(data["ids_train"])
    pid_val = pids_from_seg_ids(data["ids_val"])
    pid_test = pids_from_seg_ids(data["ids_test"])
    split_ok = (len(pid_train & pid_val) == 0
                and len(pid_train & pid_test) == 0
                and len(pid_val & pid_test) == 0
                and len(pid_train) == 85 and len(pid_val) == 18
                and len(pid_test) == 18)
    split_dir = find_first(SPLIT_DIR_CANDIDATES)
    if split_dir is not None:
        try:
            for fname, pset in [("train_participants.csv", pid_train),
                                ("validation_participants.csv", pid_val),
                                ("test_participants.csv", pid_test)]:
                import pandas as pd
                ref = set(pd.read_csv(split_dir / fname)["ID"].astype(str))
                split_ok = split_ok and (ref == pset)
        except Exception:  # noqa: BLE001
            pass

    print("\nTrain:")
    print(f"{len(X_train)} segments")
    print("\nValidation:")
    print(f"{len(X_val)} segments")
    print("\nTest:")
    print(f"{len(X_test)} segments")
    print("\nClass Distribution:")
    print(f"ADHD = {n_adhd}")
    print(f"Control = {n_ctrl}")

    # ---------- class weights (จาก TRAIN เท่านั้น) ----------
    classes = np.unique(y_train)
    try:
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight(class_weight="balanced",
                                  classes=classes, y=y_train)
        cw_source = "sklearn compute_class_weight(balanced)"
    except ImportError:
        counts = np.bincount(y_train.astype(int))
        cw = len(y_train) / (len(classes) * counts)
        cw_source = "manual balanced formula (identical to sklearn)"
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    cw_ok = all(np.isfinite(list(class_weight.values()))) and \
        all(v > 0 for v in class_weight.values())

    print("\nClass Weights:")
    print(f"Control = {class_weight[0]:.6f}")
    print(f"ADHD = {class_weight[1]:.6f}")
    print(f"(source: {cw_source})")

    # ---------- load model (ไม่ใช้ optimizer state เดิม) ----------
    untrained_path = find_first(UNTRAINED_CANDIDATES)
    if untrained_path is None:
        print("ERROR: ไม่พบ day7_03_cnn_baseline_untrained.keras")
        sys.exit(1)
    model = keras.models.load_model(untrained_path, compile=False)
    model_ok = model.input_shape == (None,) + INPUT_SHAPE

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    compile_ok = True

    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nontrain_params = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))

    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    print("\nInput:")
    print(INPUT_SHAPE)
    print("\nParameters:")
    print(total_params)

    # ---------- callbacks ----------
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH), monitor="val_loss",
        mode="min", save_best_only=True)

    # ---------- train ----------
    w_before = model.layers[0].get_weights()[0].copy()
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    t0 = time.time()
    history = model.fit(
        Xc_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        validation_data=(Xc_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stop, checkpoint],
    )
    train_time = time.time() - t0
    hist = history.history
    epochs_done = len(hist["loss"])
    trained_ok = epochs_done > 0
    w_after = model.layers[0].get_weights()[0]
    weights_changed = not np.allclose(w_before, w_after)

    # ---------- best epoch ----------
    best_idx = int(np.argmin(hist["val_loss"]))
    best_epoch = best_idx + 1
    best_val_loss = float(hist["val_loss"][best_idx])
    best_val_acc = float(hist["val_accuracy"][best_idx])
    best_tr_loss = float(hist["loss"][best_idx])
    best_tr_acc = float(hist["accuracy"][best_idx])
    use_auc = "auc" in hist
    best_auc = float(hist["auc"][best_idx]) if use_auc else None
    best_val_auc = float(hist["val_auc"][best_idx]) if use_auc else None

    print("\n" + "-" * 60)
    print("BEST MODEL")
    print("-" * 60)
    print("\nBest Epoch:")
    print(best_epoch)
    print("\nBest Validation Loss:")
    print(round(best_val_loss, 6))
    print("\nBest Validation Accuracy:")
    print(round(best_val_acc, 6))

    # ---------- finite checks ----------
    def finite(key):
        return bool(np.all(np.isfinite(hist[key])))
    hist_finite = all(finite(k) for k in
                      ["loss", "accuracy", "val_loss", "val_accuracy"])
    nan_ok = not any(np.isnan(hist[k]).any() for k in hist)
    inf_ok = not any(np.isinf(hist[k]).any() for k in hist)

    # ---------- save history csv (สะอาด) ----------
    hist_saved = False
    try:
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv_lib.writer(f)
            header = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy"]
            if use_auc:
                header += ["auc", "val_auc"]
            w.writerow(header)
            for i in range(epochs_done):
                row = [i + 1,
                       round(float(hist["loss"][i]), 6),
                       round(float(hist["accuracy"][i]), 6),
                       round(float(hist["val_loss"][i]), 6),
                       round(float(hist["val_accuracy"][i]), 6)]
                if use_auc:
                    row += [round(float(hist["auc"][i]), 6),
                            round(float(hist["val_auc"][i]), 6)]
                w.writerow(row)
        hist_saved = True
    except Exception as exc:  # noqa: BLE001
        print("ERROR save history:", exc)

    # ---------- class weights csv ----------
    with open(CLASS_WEIGHT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f)
        w.writerow(["Class", "Class_Name", "Count", "Weight"])
        w.writerow([0, "Control", n_ctrl, round(class_weight[0], 6)])
        w.writerow([1, "ADHD", n_adhd, round(class_weight[1], 6)])

    # ---------- best model saved + reload ----------
    best_saved = BEST_MODEL_PATH.exists()
    reload_ok = False
    reload_range_ok = False
    if best_saved:
        try:
            reloaded = keras.models.load_model(BEST_MODEL_PATH, compile=False)
            s = np.ascontiguousarray(X_train[:1].transpose(0, 2, 3, 1))
            p = reloaded(s, training=False).numpy()
            reload_ok = (reloaded.input_shape == (None,) + INPUT_SHAPE
                         and p.shape == (1, 1))
            reload_range_ok = bool(np.all((p >= 0.0) & (p <= 1.0)))
        except Exception as exc:  # noqa: BLE001
            print("ERROR reload best model:", exc)

    # ---------- training curve ----------
    curve_ok = False
    try:
        epochs_x = np.arange(1, epochs_done + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, hist["loss"], label="Training Loss", marker="o")
        axes[0].plot(epochs_x, hist["val_loss"], label="Validation Loss", marker="s")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(epochs_x, hist["accuracy"], label="Training Accuracy", marker="o")
        axes[1].plot(epochs_x, hist["val_accuracy"], label="Validation Accuracy", marker="s")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True)
        fig.suptitle("Day 7.4 - CNN Training Curves (ADHD screening experiment)",
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(CURVE_PNG, dpi=150, bbox_inches="tight")
        plt.close(fig)
        curve_ok = True
    except Exception as exc:  # noqa: BLE001
        print("ERROR save curve:", exc)

    # ---------- tests ----------
    tests = [
        ("Dataset loaded", ds_ok),
        ("Train shape", tr_ok),
        ("Validation shape", va_ok),
        ("Test shape", te_ok),
        ("Label validation", label_ok),
        ("Class weight validation", cw_ok),
        ("Participant split preserved", split_ok),
        ("Model loaded", model_ok),
        ("Model compiled", compile_ok),
        ("Training performed", trained_ok and weights_changed),
        ("Training history saved", hist_saved),
        ("Best model saved", best_saved),
        ("Training curve saved", curve_ok),
        ("No NaN in training history", nan_ok),
        ("No Inf in training history", inf_ok and hist_finite),
        ("Best model reload", reload_ok and reload_range_ok),
    ]
    all_ok = all(ok for _, ok in tests)

    # ---------- summaries ----------
    auc_lines = []
    if use_auc:
        auc_lines = [
            f"Best training AUC: {round(best_auc, 6)}",
            f"Best validation AUC: {round(best_val_auc, 6)}",
        ]
    train_lines = [
        "=" * 60,
        "DAY 7.4 - CNN TRAINING",
        "=" * 60,
        "Dataset",
        "",
        f"Train segments: {len(X_train)}",
        f"Validation segments: {len(X_val)}",
        f"Test segments: {len(X_test)} (ไม่ได้ใช้ระหว่าง training)",
        f"Train participants: {len(pid_train)}",
        f"Validation participants: {len(pid_val)}",
        f"Test participants: {len(pid_test)}",
        f"Train ADHD segments: {n_adhd}",
        f"Train Control segments: {n_ctrl}",
        f"Class weights: Control={class_weight[0]:.6f}, ADHD={class_weight[1]:.6f}",
        f"Class weight source: {cw_source}",
        "",
        "-" * 60,
        "MODEL",
        "-" * 60,
        "Architecture: CNN Baseline (Day 7.3, ไม่แก้ไข)",
        "Input: 65 x 7 x 19",
        f"Total parameters: {total_params}",
        f"Trainable parameters: {trainable_params}",
        f"Non-trainable parameters: {nontrain_params}",
        "",
        "-" * 60,
        "TRAINING",
        "-" * 60,
        "Optimizer: Adam",
        f"Learning rate: {LEARNING_RATE}",
        "Loss: binary_crossentropy",
        "Metrics: accuracy, auc",
        f"Batch size: {BATCH_SIZE}",
        f"Maximum epochs: {EPOCHS}",
        "Early stopping: Yes",
        f"Patience: {PATIENCE}",
        f"Best model: {BEST_MODEL_PATH.name}",
        f"Best epoch: {best_epoch}",
        f"Training epochs completed: {epochs_done}",
        f"Training time: {train_time:.1f} s "
        f"(~{train_time / max(epochs_done, 1):.1f} s/epoch)",
        "",
        "-" * 60,
        "RESULT",
        "-" * 60,
        f"Best training loss: {round(best_tr_loss, 6)}",
        f"Best validation loss: {round(best_val_loss, 6)}",
        f"Best training accuracy: {round(best_tr_acc, 6)}",
        f"Best validation accuracy: {round(best_val_acc, 6)}",
    ] + auc_lines + [
        "",
        "-" * 60,
        "MODEL STATUS",
        "-" * 60,
        "Training performed: YES",
        "Model status: TRAINED",
        f"Best model saved: {'YES' if best_saved else 'NO'}",
        "",
        "หมายเหตุ: นี่เป็น ADHD classification experiment / research prototype",
        "สำหรับ decision support เท่านั้น ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์",
        "=" * 60,
    ]
    TRAIN_SUMMARY.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    vlines = ["=" * 60, "DAY 7.4 - VALIDATION SUMMARY", "=" * 60]
    for name, ok in tests:
        vlines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    vlines += [
        "",
        f"Epochs completed: {epochs_done} | Best epoch: {best_epoch}",
        f"Best val_loss: {round(best_val_loss, 6)} | "
        f"Best val_acc: {round(best_val_acc, 6)}",
        "Test set: NOT used during training / model selection",
        "Final evaluation จะทำใน Day 7.5",
        f"Status: {'PASSED' if all_ok else 'FAILED'}",
        "=" * 60,
    ]
    VALID_SUMMARY.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    # ---------- terminal report ----------
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    print(f"Dataset validation: {'PASSED' if ds_ok else 'FAILED'}")
    print(f"Class weight validation: {'PASSED' if cw_ok else 'FAILED'}")
    print(f"Training history: {'PASSED' if hist_saved else 'FAILED'}")
    print(f"Best model save: {'PASSED' if best_saved else 'FAILED'}")
    print(f"Best model reload: {'PASSED' if (reload_ok and reload_range_ok) else 'FAILED'}")
    print(f"NaN validation: {'PASSED' if nan_ok else 'FAILED'}")
    print(f"Inf validation: {'PASSED' if (inf_ok and hist_finite) else 'FAILED'}")
    for name, ok in tests:
        if name not in ("Dataset loaded", "Class weight validation",
                        "Training history saved", "Best model saved",
                        "Best model reload", "No NaN in training history",
                        "No Inf in training history"):
            print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "-" * 60)
    print("STATUS")
    print("-" * 60)
    print("\nTraining performed:")
    print("YES" if trained_ok else "NO")
    print("\nModel status:")
    print("TRAINED" if all_ok else "CHECK FAILED")
    if all_ok:
        print(f"\nSaved: {BEST_MODEL_PATH}")
        print(f"Saved: {HISTORY_CSV}")
        print(f"Saved: {CLASS_WEIGHT_CSV}")
        print(f"Saved: {TRAIN_SUMMARY}")
        print(f"Saved: {VALID_SUMMARY}")
        print(f"Saved: {CURVE_PNG}")
        print("\nDAY 7.4: PASSED")
    else:
        print("\nDAY 7.4: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()