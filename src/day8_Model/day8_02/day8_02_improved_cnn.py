# ============================================================
# DAY 8.2 - IMPROVED CNN VERSION 1
# ทดลองลด overfitting / เพิ่มเสถียรภาพการ train โดยเทียบเท่า baseline:
#   - LR 0.001 -> 0.0003
#   - L2 1e-4 ที่ Dense(64)
#   - Dropout 0.30/0.30/0.40/0.40
#   - EarlyStopping patience 7 + ReduceLROnPlateau
# ห้าม: ใช้ test ระหว่าง train / เลือก hyperparam จาก test / ปรับ threshold
# Output ทั้งหมดบันทึกในโฟลเดอร์เดียวกับสคริปต์ (day8_02/)
# Run: python day8_02_improved_cnn.py
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
LEARNING_RATE = 0.0003
L2_FACTOR = 1e-4
ES_PATIENCE = 7
LR_PATIENCE = 3
LR_FACTOR = 0.5
LR_MIN = 1e-6
THRESHOLD = 0.5
CLASS_NAMES = {0: "Control", 1: "ADHD"}

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent   # .../day8_Model/day8_02
OUT_DIR = BASE_DIR                           # output อยู่โฟลเดอร์เดียวกับสคริปต์
DATA_ROOT = BASE_DIR.parent.parent           # .../Data

NPZ_CANDIDATES = [
    DATA_ROOT / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR.parent / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    DATA_ROOT.parent / "Data" / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz",
]
BASELINE_MODEL_CANDIDATES = [
    DATA_ROOT / "day7_cnn" / "day7_04" / "day7_04_best_model.keras",
    BASE_DIR.parent / "day7_04" / "day7_04_best_model.keras",
]
P75_PART_CANDIDATES = [
    DATA_ROOT / "day7_cnn" / "day7_05" / "day7_05_participant_predictions.csv",
    BASE_DIR.parent / "day7_05" / "day7_05_participant_predictions.csv",
]

OUT_MODEL = OUT_DIR / "day8_02_improved_model.keras"
OUT_HISTORY = OUT_DIR / "day8_02_training_history.csv"
OUT_TRAIN_SUM = OUT_DIR / "day8_02_training_summary.txt"
OUT_VAL_SUM = OUT_DIR / "day8_02_validation_summary.txt"
OUT_CW = OUT_DIR / "day8_02_class_weights.csv"
OUT_CURVE = OUT_DIR / "day8_02_training_curve.png"
OUT_TEST = OUT_DIR / "day8_02_test_results.csv"
OUT_PART = OUT_DIR / "day8_02_participant_predictions.csv"
OUT_CM = OUT_DIR / "day8_02_confusion_matrix.png"
OUT_CMP = OUT_DIR / "day8_02_comparison_with_baseline.csv"
OUT_REPORT = OUT_DIR / "day8_02_report.txt"
ALL_OUTPUTS = [OUT_MODEL, OUT_HISTORY, OUT_TRAIN_SUM, OUT_VAL_SUM, OUT_CW,
               OUT_CURVE, OUT_TEST, OUT_PART, OUT_CM, OUT_CMP, OUT_REPORT]


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
    """บันทึก learning rate หลังทุก epoch (วางต่อจาก ReduceLROnPlateau)"""
    def __init__(self):
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))


def build_improved_model():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    return keras.Sequential([
        keras.Input(shape=(65, 7, 19)),
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.30),
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.30),
        keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.40),
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(L2_FACTOR)),
        keras.layers.Dropout(0.40),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="improved_cnn_v1")


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
    print("DAY 8.2 - IMPROVED CNN VERSION 1")
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

    # นับ class ของ test participants จาก label ของ segment คนนั้น
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

    # ---------- pre-training summary ----------
    print(f"\nDataset: {npz_path.name}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Train ADHD segments = {n_adhd} | Control segments = {n_ctrl}")
    print(f"Class weights: Control={class_weight[0]:.6f}, "
          f"ADHD={class_weight[1]:.6f} ({cw_source})")
    print(f"LR={LEARNING_RATE} | L2={L2_FACTOR} | Dropout=0.30/0.30/0.40/0.40")
    print(f"ES patience={ES_PATIENCE} | ReduceLR factor={LR_FACTOR} "
          f"patience={LR_PATIENCE} min_lr={LR_MIN}")

    model = build_improved_model()
    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    nontrain_params = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    print(f"Parameters: total={total_params} trainable={trainable_params} "
          f"non_trainable={nontrain_params}")

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
        callbacks=callbacks,
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

    # ---------- training curve (3 plots) ----------
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
    fig.suptitle("Day 8.2 - Improved CNN Training Curves", fontsize=13)
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
    probs = best_model.predict(Xc_test, batch_size=64, verbose=0).flatten()
    out_shape_ok = probs.shape == (618,)
    range_ok = bool(np.all((probs >= 0.0) & (probs <= 1.0)))
    pred_labels = (probs >= THRESHOLD).astype(int)
    seg_m = compute_metrics(y_test, pred_labels, probs)

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
    g_out = g[["Participant_ID", "True_Class", "Num_Segments",
               "Mean_Probability", "Predicted_Class", "Correct"]].rename(
        columns={"Participant_ID": "ID"})
    g_out.to_csv(OUT_PART, index=False)

    y_part = g["True_Label"].to_numpy()
    p_part = g["Predicted_Label"].to_numpy()
    prob_part = g["Mean_Probability"].to_numpy()
    part_m = compute_metrics(y_part, p_part, prob_part)
    plot_cm(part_m["cm"],
            "Participant-level Confusion Matrix (Improved CNN, Test)", OUT_CM)

    # ---------- baseline metrics จากผล Day 7.5 (ไม่รันโมเดลเดิมซ้ำ) ----------
    p75_part = find_first(P75_PART_CANDIDATES)
    if p75_part is not None:
        b = pd.read_csv(p75_part)
        y_b = b["True_Label"].to_numpy()
        prob_b = b["Mean_Probability"].to_numpy()
        base_m = compute_metrics(y_b, (prob_b >= THRESHOLD).astype(int), prob_b)
        base_source = f"Day 7.5 predictions ({p75_part.name})"
    else:
        base_path = find_first(BASELINE_MODEL_CANDIDATES)
        if base_path is None:
            print("ERROR: ไม่พบผล Day 7.5 หรือ baseline model สำหรับเปรียบเทียบ")
            sys.exit(1)
        bm = keras.models.load_model(base_path, compile=False)
        pb = bm.predict(Xc_test, batch_size=64, verbose=0).flatten()
        gb = pd.DataFrame({"pid": pids, "y": y_test, "p": pb}).groupby("pid").agg(
            y=("y", "first"), p=("p", "mean")).reset_index()
        base_m = compute_metrics(gb["y"].to_numpy(),
                                 (gb["p"].to_numpy() >= THRESHOLD).astype(int),
                                 gb["p"].to_numpy())
        base_source = "re-evaluated baseline model (Day 7.5 files not found)"

    metric_names = ["Accuracy", "Precision", "Recall", "Specificity",
                    "F1", "ROC-AUC", "PR-AUC"]
    cmp_rows = [{"Metric": m,
                 "Baseline": round(base_m[m], 4),
                 "Improved_CNN": round(part_m[m], 4),
                 "Difference": round(part_m[m] - base_m[m], 4)}
                for m in metric_names]
    pd.DataFrame(cmp_rows).to_csv(OUT_CMP, index=False)

    # ---------- report ----------
    d_recall = part_m["Recall"] - base_m["Recall"]
    d_auc = part_m["ROC-AUC"] - base_m["ROC-AUC"]
    if d_recall > 0.02:
        recall_line = ("Improved CNN increased ADHD sensitivity compared "
                       "with the Baseline.")
    elif d_recall < -0.02:
        recall_line = "Improved CNN did not improve ADHD sensitivity."
    else:
        recall_line = ("The improvement did not produce a clear change in "
                       "ADHD sensitivity.")
    if d_auc > 0.02:
        auc_line = ("Participant-level ROC-AUC improved compared with the "
                    "Baseline.")
    elif d_auc < -0.02:
        auc_line = ("Participant-level ROC-AUC decreased compared with the "
                    "Baseline.")
    else:
        auc_line = ("Participant-level ROC-AUC remained roughly similar to "
                    "the Baseline.")
    gap_base = 0.7253 - 0.5595
    gap_imp = float(hist["accuracy"][best_idx] - hist["val_accuracy"][best_idx])

    report = f"""============================================================
DAY 8.2 - IMPROVED CNN VERSION 1 REPORT
============================================================

1. Objective
Test whether reducing overfitting and stabilizing training improves
the CNN Baseline for ADHD vs Control EEG spectrogram classification
(research prototype / screening support only).

2. What Changed from Baseline
- Learning rate: 0.001 -> 0.0003
- L2 regularization 1e-4 on Dense(64)
- Dropout: 0.25/0.25/0.30/0.30 -> 0.30/0.30/0.40/0.40
- EarlyStopping patience: 5 -> 7
- Added ReduceLROnPlateau (factor 0.5, patience 3, min_lr 1e-6)
- Same dataset, same participant split, same class weights, same seed 42

3. Training Result
Epochs completed: {epochs_done} / {EPOCHS}
Best epoch (min val_loss): {best_epoch}
Best train loss/acc: {hist['loss'][best_idx]:.4f} / {hist['accuracy'][best_idx]:.4f}
Training time: {train_time:.1f} s

4. Validation Result
Best val_loss: {hist['val_loss'][best_idx]:.4f}
Best val_accuracy: {hist['val_accuracy'][best_idx]:.4f}
Best val_auc: {hist['val_auc'][best_idx]:.4f}
Train-val accuracy gap at best epoch: {gap_imp:.4f}
(Baseline gap was about {gap_base:.4f})

5. Test Result - Segment Level
Accuracy {seg_m['Accuracy']:.4f} | Precision {seg_m['Precision']:.4f} |
Recall {seg_m['Recall']:.4f} | Specificity {seg_m['Specificity']:.4f} |
F1 {seg_m['F1']:.4f} | ROC-AUC {seg_m['ROC-AUC']:.4f} | PR-AUC {seg_m['PR-AUC']:.4f}
TP={seg_m['TP']} TN={seg_m['TN']} FP={seg_m['FP']} FN={seg_m['FN']}

6. Test Result - Participant Level
Accuracy {part_m['Accuracy']:.4f} | Precision {part_m['Precision']:.4f} |
Recall {part_m['Recall']:.4f} | Specificity {part_m['Specificity']:.4f} |
F1 {part_m['F1']:.4f} | ROC-AUC {part_m['ROC-AUC']:.4f} | PR-AUC {part_m['PR-AUC']:.4f}
TP={part_m['TP']} TN={part_m['TN']} FP={part_m['FP']} FN={part_m['FN']}

7. Comparison with Baseline (participant-level)
Metric            Baseline   Improved   Difference
""" + "\n".join(
        f"{r['Metric']:<16}{r['Baseline']:<11}{r['Improved_CNN']:<11}{r['Difference']:+.4f}"
        for r in cmp_rows) + f"""

8. Did the Improvement Help?
{recall_line}
{auc_line}
Overfitting gap at best epoch: {gap_imp:.4f} (baseline ~{gap_base:.4f}).
Judgement must consider the small test set (18 participants);
differences of a few participants can move these numbers.

9. Remaining Problems
- Small test set (18 participants, 9 ADHD) limits confidence.
- {'False negatives remain high (FN=' + str(part_m['FN']) + ').' if part_m['FN'] >= 4 else 'False negatives reduced but still present (FN=' + str(part_m['FN']) + ').'}
- Segment length (4 s) and spectrogram resolution unchanged.

10. Conclusion
This experiment is an ADHD screening / classification research prototype.
Results describe model behaviour on this dataset only and do not
indicate any clinical diagnostic capability.
============================================================
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    # ---------- training summary ----------
    OUT_TRAIN_SUM.write_text(f"""============================================================
DAY 8.2 - TRAINING SUMMARY
============================================================
LR: {LEARNING_RATE} | Optimizer: Adam | Loss: binary_crossentropy
L2 on Dense(64): {L2_FACTOR} | Dropout: 0.30/0.30/0.40/0.40
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
    tests = [
        ("Dataset loaded", True),
        ("Train shape correct", tr_ok),
        ("Validation shape correct", va_ok),
        ("Test shape correct", te_ok),
        ("No NaN", nan_ok),
        ("No Inf", inf_ok),
        ("Labels valid", labels_ok),
        ("Participant leakage = 0", leak_ok),
        ("Test participants = 18", part_count_ok),
        ("ADHD participants = 9", n_te_adhd == 9),
        ("Control participants = 9", n_te_ctrl == 9),
        ("Model output shape correct", out_shape_ok and range_ok),
        ("Test evaluated only after training", test_evaluated_after_training),
        # ไฟล์ validation summary ถูกเขียนหลังเช็กนี้ จึงยกเว้นมันออกชั่วคราว
        ("All output files exist",
         all(p.exists() for p in ALL_OUTPUTS if p != OUT_VAL_SUM)),
    ]
    all_ok = all(ok for _, ok in tests)

    vlines = ["=" * 60, "DAY 8.2 - VALIDATION SUMMARY", "=" * 60,
              f"Baseline comparison source: {base_source}",
              f"Epochs: {epochs_done} | Best epoch: {best_epoch}",
              ""]
    for name, ok in tests:
        vlines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    vlines += ["", f"Status: {'PASSED' if all_ok else 'FAILED'}", "=" * 60]
    OUT_VAL_SUM.write_text("\n".join(vlines) + "\n", encoding="utf-8")

    if not OUT_VAL_SUM.exists():
        print("ERROR: ไม่สามารถบันทึก day8_02_validation_summary.txt ได้")
        sys.exit(1)

    # ---------- terminal ----------
    print("\n" + "-" * 60)
    print("TEST RESULT (PARTICIPANT LEVEL)")
    print("-" * 60)
    for m in metric_names:
        print(f"{m}: baseline={base_m[m]:.4f} improved={part_m[m]:.4f} "
              f"diff={part_m[m] - base_m[m]:+.4f}")
    print(f"Confusion: TP={part_m['TP']} TN={part_m['TN']} "
          f"FP={part_m['FP']} FN={part_m['FN']}")

    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "-" * 60)
    print("STATUS")
    print("-" * 60)
    if all_ok:
        for p in ALL_OUTPUTS:
            print(f"Saved: {p.name}")
        print("\nDAY 8.2: PASSED")
    else:
        print("\nDAY 8.2: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()