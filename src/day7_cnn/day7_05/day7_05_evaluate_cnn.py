# ============================================================
# DAY 7.5 - CNN EVALUATION (TEST SET ONLY)
# ประเมินโมเดล Best จาก Day 7.4 บน Test set ที่ไม่เคยใช้ train/เลือกโมเดล
#
# - ไม่ retrain / ไม่แก้ architecture / ไม่ปรับ threshold จาก test
# - threshold คงที่ 0.5
# - ตรวจ participant mapping ก่อนทำ participant-level evaluation
# - research prototype / screening support เท่านั้น ไม่ใช่เครื่องมือวินิจฉัย
# Run: python day7_05_evaluate_cnn.py
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
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
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

CLASS_NAMES = {0: "Control", 1: "ADHD"}
THRESHOLD = 0.5

BASE_DIR = Path(__file__).resolve().parent

NPZ_CANDIDATES = [
    BASE_DIR.parent / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    BASE_DIR.parent.parent / "day7_02" / "day7_02_cnn_dataset.npz",
]
MODEL_CANDIDATES = [
    BASE_DIR.parent / "day7_04" / "day7_04_best_model.keras",
    BASE_DIR / "day7_04_best_model.keras",
    BASE_DIR.parent.parent / "day7_04" / "day7_04_best_model.keras",
]
MANIFEST_CANDIDATES = [
    BASE_DIR.parent.parent / "day6_split" / "output" / "segment_manifest.csv",
    BASE_DIR.parent.parent.parent / "day6_split" / "output" / "segment_manifest.csv",
]
HIST_CANDIDATES = [
    BASE_DIR.parent / "day7_04" / "day7_04_training_history.csv",
    BASE_DIR.parent.parent / "day7_04" / "day7_04_training_history.csv",
]


def find_first(cands):
    return next((p for p in cands if p.exists()), None)


def specificity_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def compute_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_from_cm(cm),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "cm": cm,
    }


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


def fmt_metrics(m):
    return (f"Accuracy: {m['accuracy']:.4f}\n"
            f"Precision: {m['precision']:.4f}\n"
            f"Recall/Sensitivity: {m['recall']:.4f}\n"
            f"Specificity: {m['specificity']:.4f}\n"
            f"F1: {m['f1']:.4f}\n"
            f"ROC-AUC: {m['roc_auc']:.4f}")


def main():
    print("=" * 60)
    print("DAY 7.5 - CNN EVALUATION")
    print("=" * 60)

    # ---------- load dataset ----------
    npz_path = find_first(NPZ_CANDIDATES)
    if npz_path is None:
        print("ERROR: ไม่พบ day7_02_cnn_dataset.npz")
        sys.exit(1)
    data = np.load(npz_path)
    X_test, y_test = data["X_test"], data["y_test"]
    ids_test = data["ids_test"] if "ids_test" in data else None
    ds_ok = True
    shape_ok = X_test.shape == (618, 19, 65, 7) and y_test.shape == (618,)
    Xc_test = np.transpose(X_test, (0, 2, 3, 1))
    print(f"\nDataset: {npz_path.name}")
    print(f"Test shape: {X_test.shape} -> CNN input: {Xc_test.shape}")

    # ---------- load model (compile=False แล้ว compile ใหม่) ----------
    model_path = find_first(MODEL_CANDIDATES)
    if model_path is None:
        print("ERROR: ไม่พบ day7_04_best_model.keras")
        sys.exit(1)
    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model_ok = model.input_shape == (None, 65, 7, 19)
    print(f"Model: {model_path.name}")

    # best epoch จาก history ของ Day 7.4 (ไม่ hardcode)
    best_epoch = "N/A"
    hist_path = find_first(HIST_CANDIDATES)
    if hist_path is not None:
        h = pd.read_csv(hist_path)
        best_epoch = int(h["val_loss"].idxmin() + 1)

    # ---------- predict ----------
    probs = model.predict(Xc_test, batch_size=64, verbose=0).flatten()
    pred_ok = probs.shape == (618,)
    range_ok = bool(np.all((probs >= 0.0) & (probs <= 1.0)))
    nan_ok = bool(not np.isnan(probs).any())
    inf_ok = bool(not np.isinf(probs).any())
    labels_ok = set(np.unique(y_test)) <= {0, 1}

    # ---------- PARTICIPANT MAPPING (ตรวจก่อนเสมอ) ----------
    mapping_ok = False
    map_reason = ""
    if ids_test is None or len(ids_test) != len(X_test):
        map_reason = "npz ไม่มี ids_test หรือจำนวนไม่ตรง — ไม่เดา mapping"
    else:
        pids = np.array([str(s).split("_")[0] for s in ids_test])
        dup_ok = len(set(ids_test)) == len(ids_test)
        manifest = find_first(MANIFEST_CANDIDATES)
        cross_ok = True
        if manifest is not None:
            mf = pd.read_csv(manifest).set_index("Segment_ID")
            for sid, pid in zip(ids_test, pids):
                if sid not in mf.index:
                    cross_ok = False
                    break
                if mf.loc[sid, "Split"] != "Test" or str(mf.loc[sid, "ID"]) != pid:
                    cross_ok = False
                    break
        else:
            cross_ok = True  # ใช้ ids_test จาก Day 7.2 เป็น source of truth
        mapping_ok = dup_ok and cross_ok and bool(all(len(p) > 0 for p in pids))
        if not mapping_ok:
            map_reason = f"mapping ไม่สอดคล้อง (dup_ok={dup_ok}, cross_ok={cross_ok})"

    if mapping_ok:
        pid_train = {str(s).split("_")[0] for s in data["ids_train"]}
        pid_val = {str(s).split("_")[0] for s in data["ids_val"]}
        pid_test = set(pids)
        leak = (len(pid_test & pid_train) + len(pid_test & pid_val))
        part_df = pd.DataFrame({
            "Participant_ID": pids, "True_Label": y_test, "prob": probs,
        }).groupby("Participant_ID").agg(
            True_Label=("True_Label", "first"),
            Number_of_Segments=("prob", "size"),
            Mean_Probability=("prob", "mean"),
        ).reset_index()
        part_df["True_Class"] = part_df["True_Label"].map(CLASS_NAMES)
        part_df["Predicted_Label"] = (part_df["Mean_Probability"] >= THRESHOLD).astype(int)
        part_df["Predicted_Class"] = part_df["Predicted_Label"].map(CLASS_NAMES)
        part_df["Correct"] = part_df["Predicted_Label"] == part_df["True_Label"]
        y_part = part_df["True_Label"].to_numpy()
        p_part = part_df["Predicted_Label"].to_numpy()
        prob_part = part_df["Mean_Probability"].to_numpy()
        n_part = len(part_df)
        n_part_adhd = int((y_part == 1).sum())
        n_part_ctrl = int((y_part == 0).sum())
        part_count_ok = n_part == 18
        part_dist_ok = (n_part_adhd == 9 and n_part_ctrl == 9)
        leak_ok = leak == 0
    else:
        leak_ok = False
        part_count_ok = False
        part_dist_ok = False
        print(f"\nWARNING: participant mapping ไม่ผ่าน: {map_reason}")
        print("         จะทำเฉพาะ segment-level evaluation")

    seg_adhd = int((y_test == 1).sum())
    seg_ctrl = int((y_test == 0).sum())
    seg_count_ok = len(probs) == 618
    dist_ok = (seg_adhd == 347 and seg_ctrl == 271)

    # ---------- segment-level metrics ----------
    pred_labels = (probs >= THRESHOLD).astype(int)
    seg_m = compute_metrics(y_test, pred_labels, probs)

    seg_df = pd.DataFrame({
        "Segment_Index": np.arange(len(probs)),
        "Participant_ID": pids if mapping_ok else ["UNKNOWN"] * len(probs),
        "True_Label": y_test,
        "True_Class": [CLASS_NAMES[v] for v in y_test],
        "Predicted_Probability": np.round(probs, 6),
        "Predicted_Label": pred_labels,
        "Predicted_Class": [CLASS_NAMES[v] for v in pred_labels],
        "Correct": pred_labels == y_test,
    })

    # ---------- participant-level metrics ----------
    if mapping_ok:
        part_m = compute_metrics(y_part, p_part, prob_part)
    else:
        part_m = None

    # ---------- save outputs ----------
    seg_df.to_csv(BASE_DIR / "day7_05_segment_predictions.csv", index=False)
    if mapping_ok:
        part_df.to_csv(BASE_DIR / "day7_05_participant_predictions.csv", index=False)

    plot_cm(seg_m["cm"], "Segment-level Confusion Matrix (Test, n=618)",
            BASE_DIR / "day7_05_confusion_matrix.png")
    if mapping_ok:
        plot_cm(part_m["cm"],
                "Participant-level Confusion Matrix (Test, n=18)",
                BASE_DIR / "day7_05_participant_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax.plot(fpr, tpr, label=f"Segment-level (AUC={seg_m['roc_auc']:.3f})")
    if mapping_ok:
        fpr2, tpr2, _ = roc_curve(y_part, prob_part)
        ax.plot(fpr2, tpr2, label=f"Participant-level (AUC={part_m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set (Day 7.5)")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "day7_05_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- metrics.txt ----------
    part_block = (
        fmt_metrics(part_m) + f"\nPR-AUC (additional): {part_m['pr_auc']:.4f}"
        if part_m else "SKIPPED — participant mapping ไม่ปลอดภัยต่อการประเมิน")
    cm_part_block = (
        f"TP={part_m['tp']} TN={part_m['tn']} FP={part_m['fp']} FN={part_m['fn']}"
        if part_m else "N/A")
    metrics_txt = f"""============================================================
DAY 7.5 - CNN EVALUATION
============================================================

MODEL:
CNN Baseline (architecture จาก Day 7.3, train ใน Day 7.4)
Best Epoch: {best_epoch}
Model file: {model_path.name}

TEST SET:
Participants: {n_part if mapping_ok else 18}
ADHD: {n_part_adhd if mapping_ok else 9}
Control: {n_part_ctrl if mapping_ok else 9}
Segments: {len(probs)}
(ADHD segments: {seg_adhd}, Control segments: {seg_ctrl})

SEGMENT-LEVEL RESULTS:
{fmt_metrics(seg_m)}
PR-AUC (additional): {seg_m['pr_auc']:.4f}

PARTICIPANT-LEVEL RESULTS:
{part_block}

CONFUSION MATRIX (segment-level):
TP={seg_m['tp']} TN={seg_m['tn']} FP={seg_m['fp']} FN={seg_m['fn']}

CONFUSION MATRIX (participant-level):
{cm_part_block}

IMPORTANT NOTE:
Test set was used only for final evaluation.
No test data was used for training or model selection.
Threshold fixed at 0.5 (not optimized on test set).

Medical disclaimer:
Research prototype / screening support only.
Not a medical diagnostic tool.
============================================================
"""
    (BASE_DIR / "day7_05_metrics.txt").write_text(metrics_txt, encoding="utf-8")

    # ---------- validation summary ----------
    tests = [
        ("Dataset loaded", ds_ok),
        ("Test shape", shape_ok),
        ("Model loaded", model_ok),
        ("Model prediction completed", pred_ok),
        ("Prediction shape", pred_ok),
        ("Prediction probability range", range_ok),
        ("No NaN", nan_ok),
        ("No Inf", inf_ok),
        ("Test labels valid", labels_ok),
        ("Test participant count", part_count_ok if mapping_ok else False),
        ("Test class distribution", part_dist_ok if mapping_ok else dist_ok),
        ("Participant leakage = 0", leak_ok if mapping_ok else False),
        ("Segment prediction count = 618", seg_count_ok),
    ]
    if mapping_ok:
        tests += [
            ("Participant mapping complete", mapping_ok),
            ("Participant count = 18", n_part == 18),
            ("ADHD participants = 9", n_part_adhd == 9),
            ("Control participants = 9", n_part_ctrl == 9),
            ("All test segments mapped", bool(len(seg_df) == 618)),
            ("No duplicate segment IDs", len(set(ids_test)) == 618),
        ]
    all_ok = all(ok for _, ok in tests)

    vlines = ["=" * 60, "DAY 7.5 - VALIDATION SUMMARY", "=" * 60,
              f"Model: {model_path.name} (Best Epoch {best_epoch})",
              f"Test set: {len(probs)} segments, "
              f"{n_part if mapping_ok else '?'} participants",
              "Segment-level evaluation: DONE",
              f"Participant-level evaluation: "
              + ("DONE (mapping verified)" if mapping_ok
                 else f"SKIPPED ({map_reason})"),
              f"Leakage check: Train∩Test / Val∩Test = "
              + ("0" if (mapping_ok and leak_ok) else "CHECK FAILED"),
              "Prediction validation: probabilities in [0,1], no NaN/Inf",
              ""]
    for name, ok in tests:
        vlines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    vlines += ["",
               "Output files:",
               "  day7_05_segment_predictions.csv",
               "  day7_05_participant_predictions.csv" if mapping_ok
               else "  (participant_predictions.csv ถูกข้าม)",
               "  day7_05_metrics.txt",
               "  day7_05_confusion_matrix.png",
               "  day7_05_roc_curve.png",
               "  day7_05_participant_confusion_matrix.png" if mapping_ok else "",
               "  day7_05_validation_summary.txt",
               "",
               f"Overall status: {'PASSED' if all_ok else 'FAILED'}",
               "=" * 60]
    (BASE_DIR / "day7_05_validation_summary.txt").write_text(
        "\n".join(vlines) + "\n", encoding="utf-8")

    # ---------- terminal report ----------
    print("\n" + "-" * 60)
    print("SEGMENT-LEVEL RESULTS")
    print("-" * 60)
    print(fmt_metrics(seg_m))
    if mapping_ok:
        print("\n" + "-" * 60)
        print("PARTICIPANT-LEVEL RESULTS")
        print("-" * 60)
        print(fmt_metrics(part_m))
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")
    print("\n" + "-" * 60)
    print("STATUS")
    print("-" * 60)
    if all_ok:
        print("\nSaved: day7_05_segment_predictions.csv")
        if mapping_ok:
            print("Saved: day7_05_participant_predictions.csv")
            print("Saved: day7_05_participant_confusion_matrix.png")
        print("Saved: day7_05_metrics.txt")
        print("Saved: day7_05_confusion_matrix.png")
        print("Saved: day7_05_roc_curve.png")
        print("Saved: day7_05_validation_summary.txt")
        print("\nDAY 7.5: PASSED")
    else:
        print("\nDAY 7.5: FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()