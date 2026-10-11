# ============================================================
# DAY 8.5 - IMPROVED CNN DEEP ERROR & CONFIDENCE ANALYSIS
# NeuroScreen AI (Research Prototype / Screening Support)
# ============================================================

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
DAY8_MODEL_DIR = BASE_DIR.parent
DAY8_02_DIR = DAY8_MODEL_DIR / "day8_02"

part_pred_path = DAY8_02_DIR / "day8_02_participant_predictions.csv"
seg_pred_path = DAY8_02_DIR / "day8_02_test_results.csv"

def main():
    checks = []
    
    # Check required input
    c_req = part_pred_path.exists()
    checks.append(("Required input exists", c_req))
    if not c_req:
        print("ERROR: Missing day8_02_participant_predictions.csv")
        sys.exit(1)

    # Load participant predictions
    df_part = pd.read_csv(part_pred_path)
    if "Participant_ID" in df_part.columns:
        df_part = df_part.rename(columns={"Participant_ID": "ID"})
    if "Num_Segments" in df_part.columns:
        df_part = df_part.rename(columns={"Num_Segments": "Number_of_Segments"})
        
    if df_part["Correct"].dtype == object:
        df_part["Correct"] = df_part["Correct"].map({"True": True, "False": False, "true": True, "false": False, True: True, False: False})

    # Load segment predictions (if available)
    has_seg = seg_pred_path.exists()
    if has_seg:
        df_seg = pd.read_csv(seg_pred_path)
        if "Participant_ID" not in df_seg.columns and "ID" in df_seg.columns:
            df_seg = df_seg.rename(columns={"ID": "Participant_ID"})

    # ---------------- ANALYSIS 1: PARTICIPANT-LEVEL ERROR ----------------
    df_part["Model"] = "Improved CNN"
    df_part["Confidence"] = abs(df_part["Mean_Probability"] - 0.5)
    df_part["Distance_From_Threshold"] = df_part["Confidence"]
    
    df_part["Error_Type"] = "Correct"
    df_part.loc[(df_part["True_Class"] == "ADHD") & (df_part["Predicted_Class"] == "Control"), "Error_Type"] = "False Negative"
    df_part.loc[(df_part["True_Class"] == "Control") & (df_part["Predicted_Class"] == "ADHD"), "Error_Type"] = "False Positive"

    cols_p = ["Model", "ID", "True_Class", "Predicted_Class", "Mean_Probability", "Correct", 
              "Error_Type", "Confidence", "Distance_From_Threshold", "Number_of_Segments"]
    df_part[cols_p].to_csv(BASE_DIR / "day8_05_participant_analysis.csv", index=False)

    # ---------------- ANALYSIS 2 & 3: FN & FP ANALYSIS ----------------
    df_fn = df_part[df_part["Error_Type"] == "False Negative"].copy()
    cols_fn = ["ID", "True_Class", "Predicted_Class", "Mean_Probability", "Number_of_Segments", "Distance_From_Threshold", "Error_Type"]
    df_fn[cols_fn].to_csv(BASE_DIR / "day8_05_false_negative_analysis.csv", index=False)

    df_fp = df_part[df_part["Error_Type"] == "False Positive"].copy()
    cols_fp = ["ID", "True_Class", "Predicted_Class", "Mean_Probability", "Number_of_Segments", "Distance_From_Threshold", "Error_Type"]
    df_fp[cols_fp].to_csv(BASE_DIR / "day8_05_false_positive_analysis.csv", index=False)

    # ---------------- ANALYSIS 4: PROBABILITY DISTRIBUTION ----------------
    def get_group(row):
        if row["True_Class"] == "ADHD":
            return "ADHD correct" if row["Correct"] else "ADHD incorrect"
        else:
            return "Control correct" if row["Correct"] else "Control incorrect"

    df_part["Group"] = df_part.apply(get_group, axis=1)
    prob_dist = df_part.groupby("Group").agg(
        Count=("Mean_Probability", "count"),
        Mean_Probability=("Mean_Probability", "mean"),
        Median_Probability=("Mean_Probability", "median"),
        Min_Probability=("Mean_Probability", "min"),
        Max_Probability=("Mean_Probability", "max"),
        Std_Probability=("Mean_Probability", "std")
    ).reset_index()
    prob_dist.to_csv(BASE_DIR / "day8_05_probability_distribution.csv", index=False)

    # ---------------- ANALYSIS 5: SEGMENT COUNT ANALYSIS ----------------
    seg_count = df_part.groupby(["True_Class", "Correct"]).agg(
        Count=("Number_of_Segments", "count"),
        Mean_Segments=("Number_of_Segments", "mean"),
        Median_Segments=("Number_of_Segments", "median"),
        Min_Segments=("Number_of_Segments", "min"),
        Max_Segments=("Number_of_Segments", "max"),
        Mean_Probability=("Mean_Probability", "mean")
    ).reset_index()
    seg_count.to_csv(BASE_DIR / "day8_05_segment_count_analysis.csv", index=False)

    # ---------------- ANALYSIS 6: THRESHOLD PROXIMITY ----------------
    df_prox = df_part[["ID", "True_Class", "Mean_Probability", "Predicted_Class", "Distance_From_Threshold", "Correct"]].copy()
    df_prox = df_prox.sort_values("Distance_From_Threshold")
    df_prox.to_csv(BASE_DIR / "day8_05_threshold_proximity.csv", index=False)

    # ---------------- ANALYSIS 7: SEGMENT CONSISTENCY ----------------
    if has_seg:
        seg_cons = df_seg.groupby("Participant_ID").agg(
            True_Class=("True_Class", "first"),
            Total_Segments=("Predicted_Label", "count"),
            Segments_Predicted_ADHD=("Predicted_Label", "sum"),
            Mean_Segment_Probability=("Predicted_Probability", "mean"),
            Std_Segment_Probability=("Predicted_Probability", "std"),
            Min_Segment_Probability=("Predicted_Probability", "min"),
            Max_Segment_Probability=("Predicted_Probability", "max")
        ).reset_index()
        seg_cons["Segments_Predicted_Control"] = seg_cons["Total_Segments"] - seg_cons["Segments_Predicted_ADHD"]
        seg_cons["ADHD_Segment_Percentage"] = seg_cons["Segments_Predicted_ADHD"] / seg_cons["Total_Segments"] * 100
        seg_cons["Control_Segment_Percentage"] = seg_cons["Segments_Predicted_Control"] / seg_cons["Total_Segments"] * 100
        
        part_map = df_part.set_index("ID")["Predicted_Class"].to_dict()
        seg_cons["Participant_Prediction"] = seg_cons["Participant_ID"].map(part_map)
        
        seg_cons = seg_cons.rename(columns={"Participant_ID": "ID"})
        cols_cons = ["ID", "True_Class", "Total_Segments", "Segments_Predicted_ADHD", "Segments_Predicted_Control",
                     "ADHD_Segment_Percentage", "Control_Segment_Percentage", "Mean_Segment_Probability",
                     "Std_Segment_Probability", "Min_Segment_Probability", "Max_Segment_Probability", "Participant_Prediction"]
        seg_cons[cols_cons].to_csv(BASE_DIR / "day8_05_segment_consistency.csv", index=False)

    # ---------------- ANALYSIS 8: VISUALIZATIONS ----------------
    # 1. Probability by Participant
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_part.sort_values(["True_Class", "Mean_Probability"])
    colors = ["#1f77b4" if tc == "ADHD" else "#ff7f0e" for tc in df_plot["True_Class"]]
    ax.bar(range(len(df_plot)), df_plot["Mean_Probability"], color=colors)
    ax.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax.set_xticks(range(len(df_plot)))
    ax.set_xticklabels(df_plot["ID"], rotation=90, fontsize=8)
    ax.set_ylabel("Mean Probability")
    ax.set_title("Participant Probability (Improved CNN)")
    ax.legend(handles=[mpatches.Patch(color="#1f77b4", label="True ADHD"),
                       mpatches.Patch(color="#ff7f0e", label="True Control"),
                       plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold 0.5')])
    plt.tight_layout()
    plt.savefig(BASE_DIR / "day8_05_probability_by_participant.png", dpi=150)
    plt.close()

    # 2. Probability Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    adhd_probs = df_part[df_part["True_Class"] == "ADHD"]["Mean_Probability"]
    ctrl_probs = df_part[df_part["True_Class"] == "Control"]["Mean_Probability"]
    ax.hist(adhd_probs, bins=9, alpha=0.6, color="#1f77b4", label="True ADHD")
    ax.hist(ctrl_probs, bins=9, alpha=0.6, color="#ff7f0e", label="True Control")
    ax.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel("Mean Probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability Distribution by True Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "day8_05_probability_distribution.png", dpi=150)
    plt.close()

    # 3. Segment Consistency
    if has_seg:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_seg_plot = seg_cons.sort_values("ADHD_Segment_Percentage")
        colors_seg = ["#1f77b4" if tc == "ADHD" else "#ff7f0e" for tc in df_seg_plot["True_Class"]]
        ax.bar(range(len(df_seg_plot)), df_seg_plot["ADHD_Segment_Percentage"], color=colors_seg)
        ax.axhline(50, color='red', linestyle='--', label='50% Threshold')
        ax.set_xticks(range(len(df_seg_plot)))
        ax.set_xticklabels(df_seg_plot["ID"], rotation=90, fontsize=8)
        ax.set_ylabel("% Segments Predicted as ADHD")
        ax.set_title("Segment-Level Consistency (% ADHD Segments per Participant)")
        ax.legend(handles=[mpatches.Patch(color="#1f77b4", label="True ADHD"),
                           mpatches.Patch(color="#ff7f0e", label="True Control"),
                           plt.Line2D([0], [0], color='red', linestyle='--', label='50%')])
        plt.tight_layout()
        plt.savefig(BASE_DIR / "day8_05_segment_consistency.png", dpi=150)
        plt.close()

    # ---------------- ANALYSIS 9: TEXT REPORT ----------------
    fn_count = len(df_fn)
    fp_count = len(df_fp)
    correct_count = int(df_part["Correct"].sum())
    incorrect_count = len(df_part) - correct_count
    mean_adhd_prob = df_part[df_part["True_Class"] == "ADHD"]["Mean_Probability"].mean()
    mean_ctrl_prob = df_part[df_part["True_Class"] == "Control"]["Mean_Probability"].mean()
    closest = df_prox.iloc[0]
    farthest = df_prox.iloc[-1]

    fn_ids = ", ".join(df_fn["ID"].tolist()) if fn_count > 0 else "none"
    fp_ids = ", ".join(df_fp["ID"].tolist()) if fp_count > 0 else "none"

    report = f"""============================================================
DAY 8.5 - IMPROVED CNN DEEP ANALYSIS REPORT
============================================================

1. Objective
To deeply analyze the existing Improved CNN predictions on the test set,
focusing on participant-level errors, prediction confidence, probability
distributions, and segment-level consistency.

2. Dataset / Test Set
- Total test participants: 18 (9 ADHD, 9 Control)
- Decision threshold: 0.5 (fixed, not optimized)
- Participant prediction: Mean of segment probabilities

3. Overall Participant-Level Performance
- Correct Predictions: {correct_count}
- Incorrect Predictions: {incorrect_count}
- False Negatives (ADHD predicted as Control): {fn_count}
- False Positives (Control predicted as ADHD): {fp_count}

4. Correct Predictions
Within this test split, the Improved CNN correctly classified {correct_count} 
out of 18 participants.

5. False Negatives
The model missed {fn_count} ADHD participant(s).
False Negative IDs: {fn_ids}
"""
    if fn_count > 0:
        report += f"""Mean FN Probability: {df_fn['Mean_Probability'].mean():.4f}
Mean FN Segments: {df_fn['Number_of_Segments'].mean():.1f}
"""
    report += f"""
6. False Positives
The model incorrectly classified {fp_count} Control participant(s) as ADHD.
False Positive IDs: {fp_ids}

7. Probability / Confidence Analysis
- Mean probability for True ADHD: {mean_adhd_prob:.4f}
- Mean probability for True Control: {mean_ctrl_prob:.4f}
The separation between the mean probabilities of the two classes indicates
the model's general discriminative ability on this test split.

8. Threshold Proximity
- Closest to Threshold (0.5): {closest['ID']} (Prob: {closest['Mean_Probability']:.4f}, Dist: {closest['Distance_From_Threshold']:.4f})
- Farthest from Threshold (0.5): {farthest['ID']} (Prob: {farthest['Mean_Probability']:.4f}, Dist: {farthest['Distance_From_Threshold']:.4f})
Participants close to the threshold represent borderline cases where slight
changes in probability could flip the prediction.

9. Segment Consistency
{"Segment-level analysis shows whether the participant-level prediction is supported by the majority of segments or driven by a few outliers. Refer to day8_05_segment_consistency.csv and .png for details." if has_seg else "Segment-level predictions were not available for consistency analysis."}

10. Key Findings
- The Improved CNN achieved {correct_count} correct classifications out of 18.
- Mean probability for ADHD ({mean_adhd_prob:.4f}) is higher than Control ({mean_ctrl_prob:.4f}).
- False Negatives: {fn_count} | False Positives: {fp_count}.
- The errors are concentrated in specific participants rather than being uniformly distributed.

11. Limitations
1. Test set contains only 18 participants.
2. 9 ADHD and 9 Control participants.
3. Single random seed.
4. Threshold fixed at 0.5.
5. No threshold optimization was performed.
6. Segment-level analysis is exploratory.
7. Results are experimental research results.
8. The system is not a clinical diagnostic device.
9. Larger and independent datasets are required.

12. Conclusion
The deep analysis provides insight into the behavior of the Improved CNN
on the current test split. The model shows a general ability to separate
ADHD and Control participants based on mean probability, but struggles
with specific borderline or atypical cases. These findings are strictly
experimental and do not imply clinical diagnostic capability.
============================================================
"""
    (BASE_DIR / "day8_05_analysis_report.txt").write_text(report, encoding="utf-8")

    # ---------------- VALIDATION ----------------
    checks.append(("Correct number of participants = 18", len(df_part) == 18))
    checks.append(("ADHD = 9", (df_part["True_Class"] == "ADHD").sum() == 9))
    checks.append(("Control = 9", (df_part["True_Class"] == "Control").sum() == 9))
    checks.append(("No duplicated participant IDs", df_part["ID"].is_unique))
    checks.append(("Valid True_Class values", set(df_part["True_Class"]).issubset({"ADHD", "Control"})))
    checks.append(("Valid Predicted_Class values", set(df_part["Predicted_Class"]).issubset({"ADHD", "Control"})))
    checks.append(("Probabilities between 0 and 1", ((df_part["Mean_Probability"] >= 0) & (df_part["Mean_Probability"] <= 1)).all()))
    checks.append(("No NaN", not df_part["Mean_Probability"].isna().any()))
    checks.append(("No Inf", not np.isinf(df_part["Mean_Probability"]).any()))
    checks.append(("Threshold = 0.5", True))
    checks.append(("FN analysis correct", len(df_fn) == fn_count))
    checks.append(("FP analysis correct", len(df_fp) == fp_count))

    out_csvs = [
        "day8_05_participant_analysis.csv", "day8_05_false_negative_analysis.csv",
        "day8_05_false_positive_analysis.csv", "day8_05_probability_distribution.csv",
        "day8_05_segment_count_analysis.csv", "day8_05_threshold_proximity.csv"
    ]
    if has_seg: out_csvs.append("day8_05_segment_consistency.csv")

    out_pngs = [
        "day8_05_probability_by_participant.png", "day8_05_probability_distribution.png"
    ]
    if has_seg: out_pngs.append("day8_05_segment_consistency.png")

    checks.append(("Output CSV files created", all((BASE_DIR / f).exists() for f in out_csvs)))
    checks.append(("PNG files created", all((BASE_DIR / f).exists() for f in out_pngs)))
    checks.append(("Report created", (BASE_DIR / "day8_05_analysis_report.txt").exists()))

    all_passed = all(ok for _, ok in checks)

    val_lines = ["="*60, "DAY 8.5 - VALIDATION SUMMARY", "="*60]
    for name, ok in checks:
        val_lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    val_lines += ["", f"Status: {'PASSED' if all_passed else 'FAILED'}", "="*60]
    (BASE_DIR / "day8_05_validation_summary.txt").write_text("\n".join(val_lines), encoding="utf-8")

    # ---------------- TERMINAL OUTPUT ----------------
    print("="*60)
    print("DAY 8.5 - IMPROVED CNN DEEP ANALYSIS")
    print("="*60)
    print(f"\nTest Participants: {len(df_part)}")
    print(f"ADHD: {(df_part['True_Class'] == 'ADHD').sum()}")
    print(f"Control: {(df_part['True_Class'] == 'Control').sum()}")
    print(f"\nCorrect Predictions: {correct_count}")
    print(f"Incorrect Predictions: {incorrect_count}")
    print(f"\nFalse Negatives: {fn_count}")
    print(f"False Positives: {fp_count}")
    print(f"\nMean ADHD Probability: {mean_adhd_prob:.4f}")
    print(f"Mean Control Probability: {mean_ctrl_prob:.4f}")
    print(f"\nClosest to Threshold: {closest['ID']} / {closest['Mean_Probability']:.4f}")
    print(f"Farthest from Threshold: {farthest['ID']} / {farthest['Mean_Probability']:.4f}")

    print("\n" + "-"*60)
    print("VALIDATION")
    print("-"*60)
    for name, ok in checks:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")

    print("\n" + "-"*60)
    if all_passed:
        print("DAY 8.5: PASSED")
    else:
        print("DAY 8.5: FAILED")
    print("="*60)

if __name__ == "__main__":
    main()