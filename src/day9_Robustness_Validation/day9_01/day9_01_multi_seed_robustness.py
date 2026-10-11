# ============================================================
# DAY 9.1 - MULTI-SEED ROBUSTNESS EXPERIMENT
# NeuroScreen AI (Research Prototype / Screening Support)
# แก้บั๊ก: ตรวจจำนวน "participants" ต่อคลาส แทนการนับ "segments"
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------
SEEDS = [42, 52, 62, 72, 82]
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
INPUT_SHAPE = (65, 7, 19)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent

NPZ_PATH = DATA_DIR / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz"

# Output files
OUT_SEED_RESULTS = BASE_DIR / "day9_01_seed_results.csv"
OUT_SUMMARY_STATS = BASE_DIR / "day9_01_summary_statistics.csv"
OUT_VAL_RESULTS = BASE_DIR / "day9_01_validation_results.csv"
OUT_TRAIN_SUMMARY = BASE_DIR / "day9_01_training_summary.csv"
OUT_METRICS_PNG = BASE_DIR / "day9_01_metrics_by_seed.png"
OUT_TRAIN_PNG = BASE_DIR / "day9_01_training_validation_summary.png"
OUT_REPORT = BASE_DIR / "day9_01_robustness_report.txt"
OUT_VAL_SUMMARY = BASE_DIR / "day9_01_validation_summary.txt"


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_improved_model():
    """สร้าง Improved CNN architecture ตาม Day 8.2"""
    return keras.Sequential([
        keras.Input(shape=INPUT_SHAPE),
        # Block 1
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.30),
        # Block 2
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.30),
        # Block 3
        keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.40),
        # Dense
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(L2_FACTOR)),
        keras.layers.Dropout(0.40),
        keras.layers.Dense(1, activation="sigmoid"),
    ])


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc = np.nan
    try:
        pr = average_precision_score(y_true, y_prob)
    except ValueError:
        pr = np.nan
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "Specificity": spec,
        "F1": f1, "ROC_AUC": roc, "PR_AUC": pr,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FPR": fpr, "FNR": fnr
    }


def main():
    print("=" * 60)
    print("DAY 9.1 - MULTI-SEED ROBUSTNESS")
    print("=" * 60)

    checks = []

    # ---------- Load dataset ----------
    if not NPZ_PATH.exists():
        print(f"ERROR: ไม่พบ dataset ที่ {NPZ_PATH}")
        sys.exit(1)
    data = np.load(NPZ_PATH)
    X_train_orig, y_train = data["X_train"], data["y_train"]
    X_val_orig, y_val = data["X_val"], data["y_val"]
    X_test_orig, y_test = data["X_test"], data["y_test"]
    ids_train, ids_val, ids_test = data["ids_train"], data["ids_val"], data["ids_test"]

    checks.append(("Dataset loaded successfully", True))
    checks.append(("Correct input shapes",
                   X_train_orig.shape == (2992, 19, 65, 7) and
                   X_val_orig.shape == (563, 19, 65, 7) and
                   X_test_orig.shape == (618, 19, 65, 7)))

    # Transpose
    Xc_train = X_train_orig.transpose(0, 2, 3, 1)
    Xc_val = X_val_orig.transpose(0, 2, 3, 1)
    Xc_test = X_test_orig.transpose(0, 2, 3, 1)

    # Participant leakage check
    pid_tr = {str(s).split("_")[0] for s in ids_train}
    pid_va = {str(s).split("_")[0] for s in ids_val}
    pid_te = {str(s).split("_")[0] for s in ids_test}
    leak = len(pid_tr & pid_va) + len(pid_tr & pid_te) + len(pid_va & pid_te)
    checks.append(("Correct participant split",
                   len(pid_tr) == 85 and len(pid_va) == 18 and len(pid_te) == 18))
    checks.append(("Participant leakage = 0", leak == 0))

    # ---------- FIX: นับจำนวน PARTICIPANTS ต่อคลาส (ไม่ใช่ segments) ----------
    tr_adhd_pids = {str(s).split("_")[0] for s, l in zip(ids_train, y_train) if l == 1}
    tr_ctrl_pids = {str(s).split("_")[0] for s, l in zip(ids_train, y_train) if l == 0}
    va_adhd_pids = {str(s).split("_")[0] for s, l in zip(ids_val, y_val) if l == 1}
    va_ctrl_pids = {str(s).split("_")[0] for s, l in zip(ids_val, y_val) if l == 0}
    te_adhd_pids = {str(s).split("_")[0] for s, l in zip(ids_test, y_test) if l == 1}
    te_ctrl_pids = {str(s).split("_")[0] for s, l in zip(ids_test, y_test) if l == 0}

    checks.append(("Train class counts correct",
                   len(tr_adhd_pids) == 43 and len(tr_ctrl_pids) == 42))
    checks.append(("Validation class counts correct",
                   len(va_adhd_pids) == 9 and len(va_ctrl_pids) == 9))
    checks.append(("Test class counts correct",
                   len(te_adhd_pids) == 9 and len(te_ctrl_pids) == 9))

    # NaN/Inf check
    checks.append(("No NaN", not (np.isnan(Xc_train).any() or np.isnan(Xc_val).any()
                                  or np.isnan(Xc_test).any())))
    checks.append(("No Inf", not (np.isinf(Xc_train).any() or np.isinf(Xc_val).any()
                                  or np.isinf(Xc_test).any())))

    # Class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    print(f"\nDataset: {NPZ_PATH.name}")
    print(f"Train: {Xc_train.shape} | Val: {Xc_val.shape} | Test: {Xc_test.shape}")
    print(f"Class weights: Control={class_weight[0]:.6f}, ADHD={class_weight[1]:.6f}")
    print(f"Seeds: {SEEDS}")

    # ---------- Train 5 seeds ----------
    seed_results = []
    val_results = []
    train_summaries = []
    all_prob_in_range = True
    n_seeds_completed = 0
    pids_test = np.array([str(s).split("_")[0] for s in ids_test])

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print('='*60)
        set_seeds(seed)

        model = build_improved_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        model_path = BASE_DIR / f"day9_01_best_model_seed{seed}.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                                              patience=LR_PATIENCE, min_lr=LR_MIN),
            keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor="val_loss",
                                            mode="min", save_best_only=True),
        ]

        print(f"\nTraining seed {seed}...")
        t0 = time.time()
        history = model.fit(
            Xc_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
            validation_data=(Xc_val, y_val), class_weight=class_weight,
            callbacks=callbacks, verbose=2,
        )
        train_time = time.time() - t0
        hist = history.history
        best_idx = int(np.argmin(hist["val_loss"]))
        best_epoch = best_idx + 1

        # Validation evaluation
        val_probs = model.predict(Xc_val, batch_size=64, verbose=0).flatten()
        val_preds = (val_probs >= THRESHOLD).astype(int)
        val_m = compute_metrics(y_val, val_preds, val_probs)
        val_results.append({
            "Seed": seed,
            "Best_Epoch": best_epoch,
            "Val_Loss": hist["val_loss"][best_idx],
            "Val_Accuracy": hist["val_accuracy"][best_idx],
            "Val_AUC": hist["val_auc"][best_idx],
            **{f"Val_{k}": v for k, v in val_m.items()
               if k in ["Accuracy", "Recall", "Specificity", "F1", "ROC_AUC"]}
        })

        # Training summary
        train_summaries.append({
            "Seed": seed,
            "Best_Epoch": best_epoch,
            "Train_Loss": hist["loss"][best_idx],
            "Train_Accuracy": hist["accuracy"][best_idx],
            "Train_AUC": hist["auc"][best_idx],
            "Val_Loss": hist["val_loss"][best_idx],
            "Val_Accuracy": hist["val_accuracy"][best_idx],
            "Val_AUC": hist["val_auc"][best_idx],
            "Train_Val_Accuracy_Gap": hist["accuracy"][best_idx] - hist["val_accuracy"][best_idx],
            "Training_Time_s": round(train_time, 1)
        })

        # Test evaluation
        test_probs = model.predict(Xc_test, batch_size=64, verbose=0).flatten()
        if not ((test_probs >= 0.0).all() and (test_probs <= 1.0).all()):
            all_prob_in_range = False
        test_preds = (test_probs >= THRESHOLD).astype(int)

        # Participant-level aggregation
        seg_df = pd.DataFrame({
            "Participant_ID": pids_test,
            "True_Label": y_test,
            "Predicted_Probability": test_probs,
        })
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

        # Save participant results
        g_out = g[["Participant_ID", "True_Class", "Num_Segments", "Mean_Probability",
                   "Predicted_Class", "Correct"]].rename(columns={"Participant_ID": "ID"})
        g_out.to_csv(BASE_DIR / f"day9_01_participant_results_seed{seed}.csv", index=False)

        y_part = g["True_Label"].to_numpy()
        p_part = g["Predicted_Label"].to_numpy()
        prob_part = g["Mean_Probability"].to_numpy()
        part_m = compute_metrics(y_part, p_part, prob_part)

        seed_results.append({
            "Seed": seed,
            **part_m,
            "Train_Accuracy": hist["accuracy"][best_idx],
            "Val_Accuracy": hist["val_accuracy"][best_idx],
        })

        print(f"\nSeed {seed} completed:")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Val loss: {hist['val_loss'][best_idx]:.4f}, "
              f"Val acc: {hist['val_accuracy'][best_idx]:.4f}")
        print(f"  Test participant-level: Acc={part_m['Accuracy']:.4f}, "
              f"Recall={part_m['Recall']:.4f}, Spec={part_m['Specificity']:.4f}")
        n_seeds_completed += 1

    checks.append(("All probabilities in [0,1]", all_prob_in_range))
    checks.append(("Exactly 5 seeds completed", n_seeds_completed == 5))
    checks.append(("Each seed has training result", len(train_summaries) == 5))
    checks.append(("Each seed has validation result", len(val_results) == 5))
    checks.append(("Each seed has test result", len(seed_results) == 5))

    # ---------- Save CSVs ----------
    df_seed = pd.DataFrame(seed_results)
    df_seed.to_csv(OUT_SEED_RESULTS, index=False)

    df_val = pd.DataFrame(val_results)
    df_val.to_csv(OUT_VAL_RESULTS, index=False)

    df_train = pd.DataFrame(train_summaries)
    df_train.to_csv(OUT_TRAIN_SUMMARY, index=False)

    # Summary statistics
    metrics_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC_AUC", "PR_AUC"]
    summary_rows = []
    for met in metrics_cols:
        vals = df_seed[met]
        summary_rows.append({
            "Metric": met,
            "Mean": round(vals.mean(), 4),
            "Std": round(vals.std(), 4),
            "Min": round(vals.min(), 4),
            "Max": round(vals.max(), 4),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_SUMMARY_STATS, index=False)
    checks.append(("Mean/std calculations valid",
                   len(df_summary) == len(metrics_cols) and not df_summary.isna().any().any()))

    # ---------- Plots ----------
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_cols))
    width = 0.15
    for i, seed in enumerate(SEEDS):
        vals = [df_seed.loc[df_seed["Seed"] == seed, met].values[0] for met in metrics_cols]
        ax.bar(x + i * width, vals, width, label=f"Seed {seed}")
    ax.set_ylabel("Score")
    ax.set_title("Participant-Level Metrics Across Seeds")
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(metrics_cols)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUT_METRICS_PNG, dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for seed in SEEDS:
        row = df_train[df_train["Seed"] == seed].iloc[0]
        axes[0].bar(f"Seed {seed}", row["Train_Accuracy"], alpha=0.6,
                    label=f"{seed} Train" if seed == SEEDS[0] else "")
        axes[0].bar(f"Seed {seed}", row["Val_Accuracy"], alpha=0.3,
                    label=f"{seed} Val" if seed == SEEDS[0] else "")
    axes[0].set_title("Train vs Val Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    for seed in SEEDS:
        row = df_train[df_train["Seed"] == seed].iloc[0]
        axes[1].bar(f"Seed {seed}", row["Train_Val_Accuracy_Gap"])
    axes[1].set_title("Train-Val Accuracy Gap")
    axes[1].set_ylabel("Gap")
    axes[1].axhline(0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_TRAIN_PNG, dpi=150)
    plt.close()

    # ---------- Report ----------
    report = f"""============================================================
DAY 9.1 - MULTI-SEED ROBUSTNESS REPORT
============================================================

1. Objective
Evaluate the robustness and reproducibility of the Improved CNN
(Day 8.2) by training it with 5 different random seeds and comparing
participant-level performance on the same test set.

2. Dataset
Source: day7_02_cnn_dataset.npz
Train: 2992 segments (85 participants: 43 ADHD / 42 Control)
Validation: 563 segments (18 participants: 9 ADHD / 9 Control)
Test: 618 segments (18 participants: 9 ADHD / 9 Control)
Participant-level split, leakage = 0.

3. Model Architecture
Improved CNN (same as Day 8.2):
- Conv2D 32 (3x3) + BN + MaxPool + Dropout 0.30
- Conv2D 64 (3x3) + BN + MaxPool + Dropout 0.30
- Conv2D 128 (3x3) + BN + GAP + Dropout 0.40
- Dense 64 (ReLU, L2 1e-4) + Dropout 0.40
- Dense 1 (sigmoid)
Total parameters: ~107K

4. Training Configuration
Adam lr=0.0003 | loss=binary_crossentropy | batch=32 | max epochs=30
EarlyStopping(val_loss, patience=7, restore_best)
ReduceLROnPlateau(val_loss, factor=0.5, patience=3, min_lr=1e-6)
Class weights: Control={class_weight[0]:.6f}, ADHD={class_weight[1]:.6f}

5. Results per Seed
"""
    for _, r in df_seed.iterrows():
        report += f"""
Seed {int(r['Seed'])}:
  Accuracy={r['Accuracy']:.4f}, Precision={r['Precision']:.4f}, Recall={r['Recall']:.4f},
  Specificity={r['Specificity']:.4f}, F1={r['F1']:.4f}, ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}
  TP={r['TP']}, TN={r['TN']}, FP={r['FP']}, FN={r['FN']}
"""

    report += f"""
6. Summary Statistics (Mean / Std / Min / Max)
"""
    for _, r in df_summary.iterrows():
        report += (f"{r['Metric']:<14} Mean={r['Mean']:.4f}  Std={r['Std']:.4f}  "
                   f"Min={r['Min']:.4f}  Max={r['Max']:.4f}\n")

    acc_std = df_summary[df_summary["Metric"] == "Accuracy"]["Std"].values[0]
    rec_std = df_summary[df_summary["Metric"] == "Recall"]["Std"].values[0]
    spec_std = df_summary[df_summary["Metric"] == "Specificity"]["Std"].values[0]

    report += f"""
7. Robustness Analysis
- Accuracy variability (Std): {acc_std:.4f}
- Recall variability (Std): {rec_std:.4f}
- Specificity variability (Std): {spec_std:.4f}
"""
    if acc_std < 0.05 and rec_std < 0.10 and spec_std < 0.05:
        report += """
The results were relatively consistent across random seeds, suggesting
that the Improved CNN exhibits reasonable robustness to initialization
variance on this dataset and test split.
"""
    else:
        report += """
There is noticeable variability across seeds, indicating that the
model's performance is sensitive to random initialization on this
dataset size. Further investigation with more seeds or larger datasets
would be needed.
"""

    report += f"""
8. Limitations
1. Test set contains only 18 participants (9 ADHD, 9 Control).
2. A single participant can noticeably change the metrics.
3. Only 5 random seeds were tested.
4. Threshold fixed at 0.5; no threshold optimization was performed.
5. Single dataset; no external validation.
6. Results are experimental research results.
7. The system is not a clinical diagnostic device.
8. Further validation on larger and independent datasets is required.

9. Conclusion
The multi-seed experiment provides evidence that the Improved CNN
shows {'reasonable' if acc_std < 0.05 and rec_std < 0.10 else 'some'} stability across different random
initializations on this test split. The model can be considered a
candidate for the next development stage, but the small test set
limits the strength of this conclusion.
============================================================
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    # ---------- Validation summary ----------
    checks.append(("All expected output files generated",
                   all((BASE_DIR / f"day9_01_participant_results_seed{s}.csv").exists()
                       for s in SEEDS)))
    checks.append(("Saved models exist",
                   all((BASE_DIR / f"day9_01_best_model_seed{s}.keras").exists()
                       for s in SEEDS)))
    checks.append(("Summary files readable",
                   OUT_SEED_RESULTS.exists() and OUT_SUMMARY_STATS.exists()
                   and OUT_REPORT.exists()))

    all_ok = all(ok for _, ok in checks)
    lines = ["=" * 60, "DAY 9.1 - VALIDATION SUMMARY", "=" * 60]
    for name, ok in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    lines += ["", f"Status: {'PASSED' if all_ok else 'FAILED'}", "=" * 60]
    OUT_VAL_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- Terminal ----------
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for _, r in df_summary.iterrows():
        print(f"{r['Metric']:<14} Mean={r['Mean']:.4f}  Std={r['Std']:.4f}  "
              f"Min={r['Min']:.4f}  Max={r['Max']:.4f}")

    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in checks:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    if all_ok:
        print(f"Seeds completed: {n_seeds_completed}")
        print(f"Seeds: {', '.join(map(str, SEEDS))}")
        print(f"Test participants: {len(pid_te)}")
        print(f"ADHD: {len(te_adhd_pids)}")
        print(f"Control: {len(te_ctrl_pids)}")
        print(f"Participant leakage: {leak}")
        print(f"NaN: 0")
        print(f"Inf: 0")
        print(f"Output files: PASSED")
        print("\nDAY 9.1: PASSED")
    else:
        print("DAY 9.1: FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()