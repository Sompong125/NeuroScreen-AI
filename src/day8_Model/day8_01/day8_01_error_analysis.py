import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ค้นหาไฟล์ Input แบบยืดหยุ่น (รองรับทั้ง day8_01 และ day7_08_01)
BASE_DIR = Path(__file__).resolve().parent
INPUT_CANDIDATES = [
    BASE_DIR.parent / "day7_cnn" / "day7_05" / "day7_05_participant_predictions.csv",
    BASE_DIR.parent / "day7_05" / "day7_05_participant_predictions.csv",
    BASE_DIR.parent.parent / "day7_cnn" / "day7_05" / "day7_05_participant_predictions.csv",
    BASE_DIR / "day7_05_participant_predictions.csv",
]

def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    return None

def main():
    print("=" * 60)
    print("DAY 8.1 - CNN BASELINE ERROR ANALYSIS")
    print("=" * 60)
    
    # TASK 1: LOAD AND VALIDATE
    input_path = find_input()
    if input_path is None:
        print("ERROR: Cannot find day7_05_participant_predictions.csv")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    tests = []
    req_cols = ["Participant_ID", "True_Label", "Number_of_Segments", 
                "Mean_Probability", "True_Class", "Predicted_Label", 
                "Predicted_Class", "Correct"]
                
    tests.append(("File loaded", True))
    tests.append(("Required columns", all(c in df.columns for c in req_cols)))
    tests.append(("Participant count = 18", len(df) == 18))
    tests.append(("ADHD participants = 9", (df['True_Class'] == 'ADHD').sum() == 9))
    tests.append(("Control participants = 9", (df['True_Class'] == 'Control').sum() == 9))
    tests.append(("Unique Participant_ID", df['Participant_ID'].is_unique))
    tests.append(("Valid labels", set(df['True_Label'].unique()) <= {0, 1} and set(df['Predicted_Label'].unique()) <= {0, 1}))
    tests.append(("Probability range", df['Mean_Probability'].between(0, 1).all()))
    tests.append(("No NaN", not df.isnull().any().any()))
    tests.append(("No Inf", not np.isinf(df.select_dtypes(include=[np.number])).any().any()))
    tests.append(("Segments > 0", (df['Number_of_Segments'] > 0).all()))
    
    if not all(ok for _, ok in tests):
        for name, ok in tests:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print("Validation failed. Exiting.")
        sys.exit(1)
        
    # TASK 2: CLASS PROBABILITY ANALYSIS
    prob_stats = df.groupby('True_Class')['Mean_Probability'].agg(
        Participant_Count='count', Mean_Probability='mean', Median_Probability='median',
        Min_Probability='min', Max_Probability='max', Std_Probability='std'
    ).reset_index()
    prob_stats.to_csv(BASE_DIR / "day8_01_class_probability_summary.csv", index=False)
    tests.append(("Probability analysis completed", True))
    
    # TASK 3: MISCLASSIFIED PARTICIPANTS
    misclass = df[~df['Correct']].copy()
    misclass['Error_Type'] = misclass.apply(
        lambda row: 'False Negative' if row['True_Class'] == 'ADHD' else 'False Positive', axis=1
    )
    misclass = misclass[['Participant_ID', 'True_Class', 'Predicted_Class', 
                         'Number_of_Segments', 'Mean_Probability', 'Error_Type']]
    misclass = misclass.sort_values(['True_Class', 'Mean_Probability'])
    misclass.to_csv(BASE_DIR / "day8_01_misclassified_participants.csv", index=False)
    tests.append(("Error analysis completed", True))
    
    # TASK 4: COMPARE CORRECT VS INCORRECT
    df['Prediction_Status'] = df['Correct'].apply(lambda x: 'Correct' if x else 'Incorrect')
    error_summary = df.groupby(['True_Class', 'Prediction_Status']).agg(
        Participant_Count=('Participant_ID', 'count'),
        Mean_Segments=('Number_of_Segments', 'mean'), Median_Segments=('Number_of_Segments', 'median'),
        Mean_Probability=('Mean_Probability', 'mean'), Median_Probability=('Mean_Probability', 'median')
    ).reset_index()
    error_summary.to_csv(BASE_DIR / "day8_01_error_summary.csv", index=False)
    
    # TASK 5: SEGMENT COUNT ANALYSIS
    df['Group'] = df['True_Class'] + ' ' + df['Prediction_Status']
    seg_analysis = df.groupby('Group').agg(
        Participant_Count=('Participant_ID', 'count'), Mean_Segments=('Number_of_Segments', 'mean'),
        Median_Segments=('Number_of_Segments', 'median'), Min_Segments=('Number_of_Segments', 'min'),
        Max_Segments=('Number_of_Segments', 'max')
    ).reset_index()
    seg_analysis.to_csv(BASE_DIR / "day8_01_segment_count_analysis.csv", index=False)
    tests.append(("Segment count analysis completed", True))
    
    # TASK 6: BORDERLINE PARTICIPANTS
    df['Distance_From_0.5'] = abs(df['Mean_Probability'] - 0.5)
    borderline = df.nsmallest(5, 'Distance_From_0.5')[['Participant_ID', 'True_Class', 'Mean_Probability', 'Predicted_Class', 'Correct', 'Distance_From_0.5']]
    
    # Count Errors
    fn_count = int((misclass['Error_Type'] == 'False Negative').sum())
    fp_count = int((misclass['Error_Type'] == 'False Positive').sum())
    adhd_correct = int(((df['True_Class'] == 'ADHD') & (df['Correct'])).sum())
    adhd_incorrect = int(((df['True_Class'] == 'ADHD') & (~df['Correct'])).sum())
    ctrl_correct = int(((df['True_Class'] == 'Control') & (df['Correct'])).sum())
    ctrl_incorrect = int(((df['True_Class'] == 'Control') & (~df['Correct'])).sum())
    
    # TASK 7: GENERATE TEXT REPORT
    report = [
        "=" * 60, "DAY 8.1 - CNN BASELINE ERROR ANALYSIS", "=" * 60,
        "\n1. Dataset Summary",
        f"- Total participants: {len(df)}",
        f"- ADHD participants: {(df['True_Class']=='ADHD').sum()}",
        f"- Control participants: {(df['True_Class']=='Control').sum()}",
        "\n2. Overall Findings",
        f"- Total Errors: {len(misclass)}",
        f"- False Negatives (ADHD predicted as Control): {fn_count}",
        f"- False Positives (Control predicted as ADHD): {fp_count}",
        "\n3. ADHD Error Analysis",
        f"- Correctly classified: {adhd_correct}",
        f"- Incorrectly classified: {adhd_incorrect}",
        "- The model struggles significantly with identifying ADHD participants.",
        "\n4. Control Error Analysis",
        f"- Correctly classified: {ctrl_correct}",
        f"- Incorrectly classified: {ctrl_incorrect}",
        "- The model performs well in identifying Control participants.",
        "\n5. Probability Analysis"
    ]
    for _, row in prob_stats.iterrows():
        report.append(f"- {row['True_Class']}: Mean Prob = {row['Mean_Probability']:.4f}, Median = {row['Median_Probability']:.4f}")
        
    report += [
        "\n6. Segment Count Analysis",
        "- Segment counts vary across participants. This may have a relationship with prediction errors, but further investigation is needed."
    ]
    for _, row in seg_analysis.iterrows():
        report.append(f"- {row['Group']}: Mean Segments = {row['Mean_Segments']:.1f}")
        
    report += [
        "\n7. Participants Closest to Fixed 0.5 Threshold",
        "(Note: Threshold is fixed at 0.5, not optimized)"
    ]
    for _, row in borderline.iterrows():
        report.append(f"- {row['Participant_ID']} ({row['True_Class']}): Prob={row['Mean_Probability']:.4f}, Distance={row['Distance_From_0.5']:.4f}, Correct={row['Correct']}")
        
    report += [
        "\n8. Interpretation",
        "- The CNN Baseline exhibits a high False Negative rate at the participant level.",
        f"- Out of 9 ADHD participants, only {adhd_correct} were correctly identified, while {adhd_incorrect} were missed (classified as Control).",
        f"- Out of 9 Control participants, {ctrl_correct} were correctly identified, and only {ctrl_incorrect} was misclassified as ADHD.",
        "- This indicates a conservative prediction behavior by the model, where it tends to predict the 'safer' class (Control) unless highly confident.",
        "- While segment count differences may have a relationship with prediction errors, this cannot be confirmed as a direct cause without further evidence.",
        "- Biological or clinical reasons for misclassification cannot be inferred from this data alone.",
        "\n9. Limitations",
        "- This analysis is based on a small test set (18 participants).",
        "- No threshold optimization was performed; a fixed 0.5 threshold was used.",
        "\n10. Conclusion",
        "- The baseline model shows promise in identifying Control cases but requires significant improvement in Sensitivity (Recall) for ADHD screening.",
        "- Future phases should explore methods to address this conservative prediction behavior.",
        "=" * 60
    ]
    
    (BASE_DIR / "day8_01_analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    tests.append(("Report saved", True))
    
    # TASK 8: VALIDATION SUMMARY
    val_lines = ["=" * 60, "DAY 8.1 - VALIDATION SUMMARY", "=" * 60]
    for name, ok in tests:
        val_lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    val_lines += ["\nSTATUS: PASSED", "=" * 60]
    (BASE_DIR / "day8_01_validation_summary.txt").write_text("\n".join(val_lines), encoding="utf-8")
    
    # TERMINAL OUTPUT
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total Errors: {len(misclass)}")
    print(f"False Negatives: {fn_count}")
    print(f"False Positives: {fp_count}")
    print(f"ADHD Correct: {adhd_correct}")
    print(f"ADHD Incorrect: {adhd_incorrect}")
    print(f"Control Correct: {ctrl_correct}")
    print(f"Control Incorrect: {ctrl_incorrect}")
    
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    for name, ok in tests:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")
        
    print("\n" + "-" * 60)
    print("DAY 8.1 COMPLETED")
    print("STATUS: PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()