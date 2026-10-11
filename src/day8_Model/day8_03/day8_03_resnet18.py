# ============================================================
# DAY 8.3 - RESNET-18 ON EEG SPECTROGRAMS
# เปรียบเทียบกับ Improved CNN (Day 8.2) บน dataset / split เดิม
#
# - ResNet-18 style ปรับสำหรับ input (65, 7, 19)
#   stem ไม่มี maxpool ใหญ่, downsample เฉพาะแกน frequency (strides (2,1))
# - weights=None (ไม่ใช้ ImageNet pretrained)
# - ห้ามใช้ Test ระหว่าง train / เลือก epoch / เลือก hyperparam / ปรับ threshold
# - research prototype / screening support เท่านั้น ไม่ใช่เครื่องมือวินิจฉัย
# - output ทุกไฟล์อยู่ในโฟลเดอร์เดียวกับสคริปต์ (day8_03/)
# Run: python day8_03_resnet18.py
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
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("ERROR: ยังไม่ได้ติดตั้ง TensorFlow ใน environment นี้")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# ---------------- seeds ----------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------- config ----------------
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
L2_FACTOR = 1e-4
ES_PATIENCE = 7
LR_PATIENCE = 3
LR_FACTOR = 0.5
LR_MIN = 1e-6
THRESHOLD = 0.5
CLASS_NAMES = {0: "Control", 1: "ADHD"}

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent   # .../day8_Model/day8_03
OUT_DIR = BASE_DIR                           # output อยู่โฟลเดอร์เดียวกับสคริปต์
DATA_ROOT = BASE_DIR.parent.parent           # .../Data

NPZ_CANDIDATES = [
    DATA_ROOT / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR.parent / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    DATA_ROOT.parent / "Data" / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz",
]
IMP82_PART_CANDIDATES = [
    BASE_DIR.parent / "day8_02" / "day8_02_participant_predictions.csv",
    DATA_ROOT / "day8_Model" / "day8_02" / "day8_02_participant_predictions.csv",
    DATA_ROOT / "day7_cnn" / "day8_02" / "day8_02_participant_predictions.csv",
]

OUT_MODEL = OUT_DIR / "day8_03_resnet18_best_model.keras"
OUT_HISTORY = OUT_DIR / "day8_03_training_history.csv"
OUT_TRAIN_SUM = OUT_DIR / "day8_03_training_summary.txt"
OUT_VAL_SUM = OUT_DIR / "day8_03_validation_summary.txt"
OUT_CW = OUT_DIR / "day8_03_class_weights.csv"
OUT_CURVE = OUT_DIR / "day8_03_training_curve.png"
OUT_TEST = OUT_DIR / "day8_03_test_results.csv"
OUT_PART = OUT_DIR / "day8_03_participant_predictions.csv"
OUT_CM = OUT_DIR / "day8_03_confusion_matrix.png"
OUT_CMP = OUT_DIR / "day8_03_comparison_with_improved_cnn.csv"
OUT_REPORT = OUT_DIR / "day8_03_report.txt"
ALL_OUTPUTS = [OUT_MODEL, OUT_HISTORY, OUT_TRAIN_SUM, OUT_VAL_SUM, OUT_CW,
               OUT_CURVE, OUT_TEST, OUT_PART, OUT_CM, OUT_CMP, OUT_REPORT]

# ผล Day 8.2 (participant-level) ใช้เป็น fallback ถ้าหาไฟล์ predictions ไม่เจอ
IMP82_FALLBACK = {
    "Accuracy": 0.8889, "Precision": 1.0000, "Recall": 0.7778,
    "Specificity": 1.0000, "F1": 0.8750, "ROC-AUC": 0.8889, "PR-AUC": 0.9233,
}


def find_first(cands):
    return next((p for p in cands if p.exists()), None)


def specificity_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": specificity_from_cm(cm),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "cm": cm,
    }


class LRRecorder(keras.callbacks.Callback):
    """บันทึก + พิมพ์ learning rate ต้นทุก epoch (ค่าที่ใช้จริงใน epoch นั้น)"""
    def __init__(self):
        self.lrs = []
    def on_epoch_begin(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        self.lrs.append(lr)
        print(f"  learning_rate: {lr:.8f}")


# ---------------- ResNet-18 แบบปรับสำหรับ EEG spectrogram ----------------
def _conv_bn_relu(x, filters, kernel, strides):
    x = keras.layers.Conv2D(filters, kernel, strides=strides,
                            padding="same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def _basic_block(x, filters, stride):
    """ResNet basic block: 3x3 -> 3x3 + shortcut (projection เมื่อจำเป็น)"""
    shortcut = x
    y = keras.layers.Conv2D(filters, 3, strides=stride,
                            padding="same", use_bias=False)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)
    y = keras.layers.Conv2D(filters, 3, strides=1,
                            padding="same", use_bias=False)(y)
    y = keras.layers.BatchNormalization()(y)
    if stride != (1, 1) or int(shortcut.shape[-1]) != filters:
        shortcut = keras.layers.Conv2D(filters, 1, strides=stride,
                                       padding="same", use_bias=False)(shortcut)
        shortcut = keras.layers.BatchNormalization()(shortcut)
    y = keras.layers.Add()([y, shortcut])
    return keras.layers.Activation("relu")(y)


def build_resnet18_eeg():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    inp = keras.Input(shape=(65, 7, 19))
    # stem: conv 64, stride 1, ไม่มี maxpool 7x7/2 แบบ ImageNet
    x = _conv_bn_relu(inp, 64, 3, (1, 1))
    # 4 groups x 2 blocks (ResNet-18)
    # downsample เฉพาะแกน frequency (65->33->17->9) เก็บแกน time = 7 ไว้
    for si, f in enumerate([64, 128, 256, 512]):
        stride = (1, 1) if si == 0 else (2, 1)
        x = _basic_block(x, f, stride)
        x = _basic_block(x, f, (1, 1))
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(L2_FACTOR))(x)
    x = keras.layers.Dropout(0.4)(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="resnet18_eeg")


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Control", "ADHD"]); ax.set_yticklabels(["Control", "ADHD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("DAY 8.3 - RESNET-18")
    print("=" * 60)

    # ---------- load dataset ----------
    npz_path = find_first(NPZ_CANDIDATES)
    if npz_path is None:
        print("ERROR: ไม่พบ day7_02_cnn_dataset.npz ใน:")
        for p in NPZ_CANDIDATES:
            print("  ", p)
        sys.exit(1)
    data = np.load(npz_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    ids_train, ids_val, ids_test = (data["ids_train"], data["ids_val"],
                                    data["ids_test"])

    tr_ok = X_train.shape == (2992, 19, 65, 7)
    va_ok = X_val.shape == (563, 19, 65, 7)
    te_ok = X_test.shape == (618, 19, 65, 7)
    nan_ok = not (np.isnan(X_train).any() or np.isnan(X_val).any()
                  or np.isnan(X_test).any())
    inf_ok = not (np.isinf(X_train).any() or np.isinf(X_val).any()
                  or np.isinf(X_test).any())
    labels_ok = (set(np.unique(y_train)) <= {0, 1}
                 and set(np.unique(y_val)) <= {0, 1}
                 and set(np.unique(y_test)) <= {0, 1})

    pid_tr = {str(s).split("_")[0] for s in ids_train}
    pid_va = {str(s).split("_")[0] for s in ids_val}
    pid_te = {str(s).split("_")[0] for s in ids_test}
    leak = (len(pid_tr & pid_va) + len(pid_tr & pid_te) + len(pid_va & pid_te))
    leak_ok = leak == 0
    part_count_ok = len(pid_te) == 18

    pid_label = {}
    for sid, lab in zip(ids_test, y_test):
        pid_label[str(sid).split("_")[0]] = int(lab)
    n_te_adhd = sum(1 for v in pid_label.values() if v == 1)
    n_te_ctrl = sum(1 for v in pid_label.values() if v == 0)
    dist_ok = (n_te_adhd == 9 and n_te_ctrl == 9)

    Xc_train = X_train.transpose(0, 2, 3, 1)
    Xc_val = X_val.transpose(0, 2, 3, 1)
    Xc_test = X_test.transpose(0, 2, 3, 1)

    n_adhd = int((y_train == 1).sum())
    n_ctrl = int((y_train == 0).sum())

    print(f"\nDataset: {npz_path.name}")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    print(f"\nInput shape: (65, 7, 19)")

    # ---------- class weights (TRAIN เท่านั้น) ----------
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
    print(f"\nClass weights: Control={class_weight[0]:.6f}, "
          f"ADHD={class_weight[1]:.6f} ({cw_source})")

    # ---------- model ----------
    model = build_resnet18_eeg()
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nontrain_params = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    print(f"\nModel parameters: total={total_params} "
          f"trainable={trainable_params} non_trainable={nontrain_params}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    lr_rec = LRRecorder()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                                          patience=LR_PATIENCE, min_lr=LR_MIN),
        keras.callbacks.ModelCheckpoint(filepath=str(OUT_MODEL),
                                        monitor="val_loss", mode="min",
                                        save_best_only=True),
        lr_rec,
    ]

    # ---------- train (ไม่มี test เข้ามาเกี่ยว) ----------
    print("\nTraining ...")
    t0 = time.time()
    history = model.fit(
        Xc_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
        validation_data=(Xc_val, y_val), class_weight=class_weight,
        callbacks=callbacks, verbose=2,
    )
    train_time = time.time() - t0
    hist = history.history
    epochs_done = len(hist["loss"])
    best_idx = int(np.argmin(hist["val_loss"]))
    best_epoch = best_idx + 1

    # ---------- history csv ----------
    with open(OUT_HISTORY, "w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f)
        w.writerow(["epoch", "loss", "accuracy", "auc", "val_loss",
                    "val_accuracy", "val_auc", "learning_rate"])
        for i in range(epochs_done):
            w.writerow([i + 1, round(float(hist["loss"][i]), 6),
                        round(float(hist["accuracy"][i]), 6),
                        round(float(hist["auc"][i]), 6),
                        round(float(hist["val_loss"][i]), 6),
                        round(float(hist["val_accuracy"][i]), 6),
                        round(float(hist["val_auc"][i]), 6),
                        round(lr_rec.lrs[i], 8)])

    # ---------- training curve ----------
    ex = np.arange(1, epochs_done + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ex, hist["loss"], label="Train Loss", marker="o")
    axes[0].plot(ex, hist["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(ex, hist["accuracy"], label="Train Accuracy", marker="o")
    axes[1].plot(ex, hist["val_accuracy"], label="Val Accuracy", marker="s")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(True)
    axes[2].plot(ex, hist["auc"], label="Train AUC", marker="o")
    axes[2].plot(ex, hist["val_auc"], label="Val AUC", marker="s")
    axes[2].set_title("AUC"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC"); axes[2].legend(); axes[2].grid(True)
    fig.suptitle("Day 8.3 - ResNet-18 Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- class weights csv ----------
    with open(OUT_CW, "w", newline="", encoding="utf-8") as f:
        w = csv_lib.writer(f)
        w.writerow(["Class", "Class_Name", "Count", "Weight"])
        w.writerow([0, "Control", n_ctrl, round(class_weight[0], 6)])
        w.writerow([1, "ADHD", n_adhd, round(class_weight[1], 6)])

    # ---------- FINAL TEST EVALUATION (ครั้งเดียว หลัง train จบ) ----------
    test_evaluated_after_training = True
    best_model = keras.models.load_model(OUT_MODEL, compile=False)
    raw_out = best_model.predict(Xc_test, batch_size=64, verbose=0)
    out_shape_ok = raw_out.shape == (len(X_test), 1)
    probs = raw_out.flatten()
    range_ok = bool(np.all((probs >= 0.0) & (probs <= 1.0)))
    pred_labels = (probs >= THRESHOLD).astype(int)
    seg_m = compute_metrics(y_test, pred_labels, probs)

    print("\n" + "=" * 60)
    print("TEST RESULT (SEGMENT LEVEL)")
    print("=" * 60)
    for k in ["Accuracy", "Precision", "Recall", "Specificity", "F1",
              "ROC-AUC", "PR-AUC"]:
        print(f"{k}: {seg_m[k]:.4f}")
    print(f"TP={seg_m['TP']} TN={seg_m['TN']} FP={seg_m['FP']} FN={seg_m['FN']}")

    pids = np.array([str(s).split("_")[0] for s in ids_test])
    seg_df = pd.DataFrame({
        "Segment_Index": np.arange(len(probs)),
        "Participant_ID": pids,
        "True_Label": y_test,
        "True_Class": [CLASS_NAMES[v] for v in y_test],
        "Predicted_Probability": np.round(probs, 6),
        "Predicted_Label": pred_labels,
        "Predicted_Class": [CLASS_NAMES[v] for v in pred_labels],
        "Correct": pred_labels == y_test,
    })
    seg_df.to_csv(OUT_TEST, index=False)

    # ---------- participant-level ----------
    g = seg_df.groupby("Participant_ID").agg(
        True_Label=("True_Label", "first"),
        Num_Segments=("Predicted_Probability", "size"),
        Mean_Probability=("Predicted_Probability", "mean"),
    ).reset_index()
    g["Mean_Probability"] = g["Mean_Probability"].round(6)
    g["True_Class"] = g["True_Label"].map(CLASS_NAMES)
    g["Predicted_Label"] = (g["Mean_Probability"] >= THRESHOLD).astype(int)
    g["Predicted_Class"] = g["Predicted_Label"].map(CLASS_NAMES)
    g["Correct"] = g["Predicted_Label"] == g["True_Label"]
    g[["Participant_ID", "True_Class", "Num_Segments", "Mean_Probability",
       "Predicted_Class", "Correct"]].rename(
        columns={"Participant_ID": "ID"}).to_csv(OUT_PART, index=False)

    y_part = g["True_Label"].to_numpy()
    p_part = g["Predicted_Label"].to_numpy()
    prob_part = g["Mean_Probability"].to_numpy()
    part_m = compute_metrics(y_part, p_part, prob_part)
    plot_cm(part_m["cm"],
            "Participant-level Confusion Matrix (ResNet-18, Test)", OUT_CM)

    print("\n" + "=" * 60)
    print("TEST RESULT (PARTICIPANT LEVEL)")
    print("=" * 60)
    for k in ["Accuracy", "Precision", "Recall", "Specificity", "F1",
              "ROC-AUC", "PR-AUC"]:
        print(f"{k}: {part_m[k]:.4f}")
    print(f"TP={part_m['TP']} TN={part_m['TN']} FP={part_m['FP']} FN={part_m['FN']}")

    # ---------- error analysis (คอลัมน์ใน g คือ Participant_ID) ----------
    fn_ids = g[(g["True_Class"] == "ADHD") &
               (g["Predicted_Class"] == "Control")]["Participant_ID"].tolist()
    fp_ids = g[(g["True_Class"] == "Control") &
               (g["Predicted_Class"] == "ADHD")]["Participant_ID"].tolist()
    adhd_ok = int(((g["True_Class"] == "ADHD") & g["Correct"]).sum())
    adhd_bad = int(((g["True_Class"] == "ADHD") & ~g["Correct"]).sum())
    ctrl_ok = int(((g["True_Class"] == "Control") & g["Correct"]).sum())
    ctrl_bad = int(((g["True_Class"] == "Control") & ~g["Correct"]).sum())

    # ---------- comparison กับ Improved CNN (Day 8.2) ----------
    imp_path = find_first(IMP82_PART_CANDIDATES)
    if imp_path is not None:
        b = pd.read_csv(imp_path)
        y_b = b["True_Class"].map({"ADHD": 1, "Control": 0}).to_numpy()
        prob_b = b["Mean_Probability"].to_numpy()
        imp_m = compute_metrics(y_b, (prob_b >= THRESHOLD).astype(int), prob_b)
        imp_source = f"Day 8.2 predictions ({imp_path.name})"
    else:
        imp_m = dict(IMP82_FALLBACK)
        imp_source = "Day 8.2 reported results (fallback constants)"

    metric_names = ["Accuracy", "Precision", "Recall", "Specificity",
                    "F1", "ROC-AUC", "PR-AUC"]
    cmp_rows = [{"Metric": m,
                 "Improved_CNN": round(imp_m[m], 4),
                 "ResNet18": round(part_m[m], 4),
                 "Difference": round(part_m[m] - imp_m[m], 4)}
                for m in metric_names]
    pd.DataFrame(cmp_rows).to_csv(OUT_CMP, index=False)

    print("\n" + "=" * 60)
    print("COMPARISON WITH IMPROVED CNN (participant-level)")
    print("=" * 60)
    print(f"Source: {imp_source}")
    print(f"{'Metric':<14}{'Improved_CNN':>14}{'ResNet18':>12}{'Difference':>12}")
    for r in cmp_rows:
        print(f"{r['Metric']:<14}{r['Improved_CNN']:>14.4f}"
              f"{r['ResNet18']:>12.4f}{r['Difference']:>+12.4f}")

    # ---------- report ----------
    gap_imp = float(hist["accuracy"][best_idx] - hist["val_accuracy"][best_idx])
    report = f"""============================================================
DAY 8.3 - RESNET-18 REPORT
============================================================

1. Objective
Evaluate a ResNet-18 style CNN on the same EEG spectrogram dataset and
participant split used in previous experiments, and compare its test
behaviour with the Improved CNN from Day 8.2.

2. Dataset
Source: day7_02_cnn_dataset.npz (created in Day 7.2, unchanged)
Train: 2992 segments (85 participants: 43 ADHD / 42 Control)
Validation: 563 segments (18 participants: 9 ADHD / 9 Control)
Test: 618 segments (18 participants: 9 ADHD / 9 Control)
Participant-level split from Day 6 (seed 42), leakage = 0.

3. Input Shape
Original: (N, 19, 65, 7) -> transposed to (N, 65, 7, 19)
65 = frequency bins, 7 = time bins, 19 = EEG channels.

4. ResNet-18 Architecture
Stem: Conv2D 64 (3x3, stride 1, same) + BN + ReLU (no ImageNet maxpool)
4 stages x 2 basic blocks: 64 / 128 / 256 / 512 filters
Downsampling stride (2,1): frequency axis only (65->33->17->9),
time axis kept at 7 to avoid collapsing the narrow time dimension.
Projection shortcuts (1x1 conv) where filters/stride change.
GlobalAveragePooling2D -> Dropout 0.4 -> Dense 64 (ReLU, L2 1e-4)
-> Dropout 0.4 -> Dense 1 sigmoid. weights=None (no ImageNet).
Parameters: total={total_params}, trainable={trainable_params},
non_trainable={nontrain_params}

5. Training Configuration
Adam lr=0.0001 | loss=binary_crossentropy | metrics=accuracy, AUC
Batch 32 | max epochs 30 | shuffle | seed 42
Class weights from TRAIN labels only:
Control={class_weight[0]:.6f}, ADHD={class_weight[1]:.6f}
EarlyStopping(val_loss, patience=7, restore_best_weights)
ReduceLROnPlateau(val_loss, factor=0.5, patience=3, min_lr=1e-6)
ModelCheckpoint(val_loss, save_best_only)

6. Training Result
Epochs completed: {epochs_done} / {EPOCHS}
Best epoch (min val_loss): {best_epoch}
Best train loss/acc: {hist['loss'][best_idx]:.4f} / {hist['accuracy'][best_idx]:.4f}
Training time: {train_time:.1f} s

7. Validation Result
Best val_loss: {hist['val_loss'][best_idx]:.4f}
Best val_accuracy: {hist['val_accuracy'][best_idx]:.4f}
Best val_auc: {hist['val_auc'][best_idx]:.4f}
Train-val accuracy gap at best epoch: {gap_imp:.4f}

8. Test Result - Segment Level
Accuracy {seg_m['Accuracy']:.4f} | Precision {seg_m['Precision']:.4f} |
Recall {seg_m['Recall']:.4f} | Specificity {seg_m['Specificity']:.4f} |
F1 {seg_m['F1']:.4f} | ROC-AUC {seg_m['ROC-AUC']:.4f} | PR-AUC {seg_m['PR-AUC']:.4f}
TP={seg_m['TP']} TN={seg_m['TN']} FP={seg_m['FP']} FN={seg_m['FN']}

9. Test Result - Participant Level
Accuracy {part_m['Accuracy']:.4f} | Precision {part_m['Precision']:.4f} |
Recall {part_m['Recall']:.4f} | Specificity {part_m['Specificity']:.4f} |
F1 {part_m['F1']:.4f} | ROC-AUC {part_m['ROC-AUC']:.4f} | PR-AUC {part_m['PR-AUC']:.4f}
TP={part_m['TP']} TN={part_m['TN']} FP={part_m['FP']} FN={part_m['FN']}

10. Comparison with Improved CNN (participant-level, same test set)
Source: {imp_source}
Metric            Improved_CNN   ResNet18     Difference
""" + "\n".join(
        f"{r['Metric']:<16}{r['Improved_CNN']:<15}{r['ResNet18']:<13}{r['Difference']:+.4f}"
        for r in cmp_rows) + f"""

No model is declared a winner from a single test set of 18 participants.
The table reports observed behaviour on this test split only.

11. Error Analysis (participant-level)
ADHD correct: {adhd_ok} | ADHD incorrect: {adhd_bad}
Control correct: {ctrl_ok} | Control incorrect: {ctrl_bad}
False Negative participants: {', '.join(fn_ids) if fn_ids else 'none'}
False Positive participants: {', '.join(fp_ids) if fp_ids else 'none'}
Note: misclassification causes cannot be inferred as biological or
clinical from this experiment; further investigation would be needed.

12. Limitations
- Small test set (18 participants, 9 ADHD): a few participants move metrics a lot.
- Single seed / single run; no repeated-seed confidence estimates.
- 4-second segments and fixed spectrogram settings unchanged.
- Validation curve noise affects best-epoch selection.

13. Conclusion
ResNet-18 was evaluated on the same EEG spectrogram dataset and
participant split used in previous experiments. The results describe
model behavior on this dataset and test set. They should not be
interpreted as clinical diagnostic performance. This system remains a
research prototype / screening support tool, not a medical device.
============================================================
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    # ---------- training summary ----------
    OUT_TRAIN_SUM.write_text(f"""============================================================
DAY 8.3 - TRAINING SUMMARY
============================================================
Model: ResNet-18 (EEG-adapted, weights=None)
LR: {LEARNING_RATE} | Optimizer: Adam | Loss: binary_crossentropy
L2 on Dense(64): {L2_FACTOR} | Dropout: 0.4 / 0.4
Batch: {BATCH_SIZE} | Max epochs: {EPOCHS} | Seed: {RANDOM_SEED}
Class weights (train only): Control={class_weight[0]:.6f}, ADHD={class_weight[1]:.6f}
Callbacks: EarlyStopping(val_loss, patience={ES_PATIENCE}, restore_best),
           ReduceLROnPlateau(val_loss, factor={LR_FACTOR}, patience={LR_PATIENCE}, min_lr={LR_MIN}),
           ModelCheckpoint(val_loss, save_best_only)
Parameters: total={total_params} trainable={trainable_params} non_trainable={nontrain_params}
Epochs completed: {epochs_done} | Best epoch: {best_epoch}
Best val_loss: {hist['val_loss'][best_idx]:.6f}
Best val_accuracy: {hist['val_accuracy'][best_idx]:.6f}
Best val_auc: {hist['val_auc'][best_idx]:.6f}
Training time: {train_time:.1f} s
============================================================
""", encoding="utf-8")

    # ---------- validation checks ----------
    best_saved = OUT_MODEL.exists()
    tests = [
        ("Dataset loaded", True),
        ("Train shape", tr_ok),
        ("Validation shape", va_ok),
        ("Test shape", te_ok),
        ("No NaN", nan_ok),
        ("No Inf", inf_ok),
        ("Labels valid", labels_ok),
        ("Participant leakage = 0", leak_ok),
        ("Test participants = 18", part_count_ok),
        ("ADHD participants = 9", n_te_adhd == 9),
        ("Control participants = 9", n_te_ctrl == 9),
        ("Model output shape = (N,1)", out_shape_ok and range_ok),
        ("Test evaluated only after training", test_evaluated_after_training),
        ("Best model saved", best_saved),
        # ไฟล์ validation summary ถูกเขียนหลังเช็กนี้ จึงยกเว้นชั่วคราว
        ("All output files exist",
         all(p.exists() for p in ALL_OUTPUTS if p != OUT_VAL_SUM)),
    ]
    all_ok = all(ok for _, ok in tests)

    vlines = ["=" * 60, "DAY 8.3 - VALIDATION SUMMARY", "=" * 60,
              f"Comparison source: {imp_source}",
              f"Epochs: {epochs_done} | Best epoch: {best_epoch}",
              ""]
    for name, ok in tests:
        vlines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    vlines += ["", f"Status: {'PASSED' if all_ok else 'FAILED'}", "=" * 60]
    OUT_VAL_SUM.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    if not OUT_VAL_SUM.exists():
        print("ERROR: ไม่สามารถบันทึก day8_03_validation_summary.txt ได้")
        sys.exit(1)

    # ---------- terminal ----------
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print("STATUS")
    print("=" * 60)
    if all_ok:
        for p in ALL_OUTPUTS:
            print(f"Saved: {p.name}")
        print("\nDAY 8.3: PASSED")
    else:
        print("\nDAY 8.3: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()