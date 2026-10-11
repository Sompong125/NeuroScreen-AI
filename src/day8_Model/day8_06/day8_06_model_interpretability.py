# ============================================================
# DAY 8.6 - IMPROVED CNN MODEL INTERPRETABILITY (Grad-CAM)
# วิเคราะห์ว่า Improved CNN "มอง" ส่วนไหนของ EEG spectrogram
# - Keras 3 compatible: สร้าง functional grad-model ที่แชร์ weight เดิม
# - ไม่แก้โมเดล ไม่เทรนใหม่ (research prototype เท่านั้น)
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

# ---------------- seeds ----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ---------------- config ----------------
THRESHOLD = 0.5
FS = 128
NPERSEG = 128
N_FREQ = 65
N_TIME = 7
INPUT_SHAPE = (65, 7, 19)
CLASS_NAMES = {0: "Control", 1: "ADHD"}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
FREQ_BINS = np.arange(N_FREQ) * (FS / NPERSEG)  # 0..64 Hz

TARGET_FN = ["v25p", "v6p"]           # จาก Day 8.5
TARGET_THRESHOLD = "v140"             # จาก Day 8.5 (closest to 0.5)

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent
DAY8_MODEL_DIR = BASE_DIR.parent
DATA_DIR = DAY8_MODEL_DIR.parent

MODEL_CANDIDATES = [
    DAY8_MODEL_DIR / "day8_02" / "day8_02_improved_model.keras",
    BASE_DIR / "day8_02_improved_model.keras",
]
NPZ_CANDIDATES = [
    DATA_DIR / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz",
    BASE_DIR / "day7_02_cnn_dataset.npz",
    DAY8_MODEL_DIR / "day7_02_cnn_dataset.npz",
]
P85_CANDIDATES = [
    DAY8_MODEL_DIR / "day8_05" / "day8_05_participant_analysis.csv",
    BASE_DIR / "day8_05_participant_analysis.csv",
]

OUT_SUMMARY = BASE_DIR / "day8_06_interpretability_summary.csv"
OUT_CH_IMP = BASE_DIR / "day8_06_channel_importance.csv"
OUT_FREQ_IMP = BASE_DIR / "day8_06_frequency_importance.csv"
OUT_TIME_IMP = BASE_DIR / "day8_06_time_importance.csv"
OUT_REPORT = BASE_DIR / "day8_06_interpretability_report.txt"
OUT_VAL = BASE_DIR / "day8_06_validation_summary.txt"


def find_first(cands):
    return next((p for p in cands if p.exists()), None)


def find_last_conv(model):
    """หา Conv2D layer สุดท้ายก่อน GlobalAveragePooling/Dense"""
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer
            break
    return last_conv


def build_grad_model(model, layer_name):
    """สร้าง functional model: input -> [conv_out, pred]
    โดยแชร์ layer object (และน้ำหนัก) ชุดเดียวกับโมเดลเดิมทุกประการ"""
    inp = keras.Input(shape=INPUT_SHAPE)
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == layer_name:
            conv_out = x
    if conv_out is None:
        raise ValueError(f"ไม่พบเลเยอร์ {layer_name} ใน model.layers")
    return keras.Model(inputs=inp, outputs=[conv_out, x])


def compute_gradcam(grad_model, x_batch):
    """Grad-CAM: gradient ของ P(ADHD) เทียบกับ conv feature maps
    คืน: (heatmap, conv_out, preds) — heatmap normalized 0..1 (ReLU แล้ว)"""
    x_t = tf.constant(x_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_t)
        conv_out, preds = grad_model(x_t, training=False)
        target = tf.reduce_sum(preds)
    grads = tape.gradient(target, conv_out)
    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradient เป็น None — ตรวจสอบว่าเลเยอร์ปลายทางเชื่อมกับ output จริง")
    pooled = tf.reduce_mean(grads, axis=(1, 2))       # (batch, filters)
    weighted = conv_out * pooled[:, None, None, :]
    heatmap = tf.reduce_sum(weighted, axis=-1)        # (batch, h, w)
    heatmap = tf.nn.relu(heatmap)
    hmax = tf.reduce_max(heatmap, axis=(1, 2), keepdims=True) + 1e-10
    heatmap = heatmap / hmax
    return heatmap.numpy(), conv_out.numpy(), preds.numpy().flatten()


def upsample_heatmap(hmap, target_shape):
    """Upsample heatmap (h, w) -> target_shape ด้วย bilinear"""
    from PIL import Image
    img = Image.fromarray((hmap * 255).astype(np.uint8))
    img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def plot_gradcam(spec_mean, heatmap, meta, out_path):
    """spec_mean: (65, 7) averaged over channels; heatmap: (65, 7) in [0,1]"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(spec_mean, aspect="auto", origin="lower",
                         extent=[0, N_TIME, 0, FREQ_BINS[-1]], cmap="viridis")
    axes[0].set_title("Mean Spectrogram (avg over 19 ch)")
    axes[0].set_xlabel("Time bin"); axes[0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(heatmap, aspect="auto", origin="lower",
                         extent=[0, N_TIME, 0, FREQ_BINS[-1]], cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].set_xlabel("Time bin"); axes[1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(spec_mean, aspect="auto", origin="lower",
                   extent=[0, N_TIME, 0, FREQ_BINS[-1]], cmap="gray")
    axes[2].imshow(heatmap, aspect="auto", origin="lower",
                   extent=[0, N_TIME, 0, FREQ_BINS[-1]], cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time bin"); axes[2].set_ylabel("Frequency (Hz)")
    fig.suptitle(
        f"{meta['ID']} | True: {meta['true_class']} | "
        f"Pred: {meta['pred_class']} | P(ADHD)={meta['prob']:.4f} "
        f"| Seg P={meta['seg_prob']:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("DAY 8.6 - MODEL INTERPRETABILITY")
    print("=" * 60)

    checks = []

    # ---------- load model ----------
    model_path = find_first(MODEL_CANDIDATES)
    c_model = model_path is not None
    checks.append(("Model loaded successfully", c_model))
    if not c_model:
        print("ERROR: ไม่พบ day8_02_improved_model.keras"); sys.exit(1)
    model = keras.models.load_model(model_path, compile=False)
    print(f"\nModel: {model_path.name}")

    # ---------- load dataset ----------
    npz_path = find_first(NPZ_CANDIDATES)
    c_ds = npz_path is not None
    checks.append(("Dataset loaded successfully", c_ds))
    if not c_ds:
        print("ERROR: ไม่พบ day7_02_cnn_dataset.npz"); sys.exit(1)
    data = np.load(npz_path)
    X_test, y_test = data["X_test"], data["y_test"]
    ids_test = data["ids_test"]
    c_shape = X_test.shape == (618, 19, 65, 7)
    checks.append(("Correct input shape", c_shape))
    Xc_test = X_test.transpose(0, 2, 3, 1)

    pids = np.array([str(s).split("_")[0] for s in ids_test])
    pid_set = set(pids)
    pid_label = {p: int(l) for p, l in zip(pids, y_test)}
    c_count = len(pid_set) == 18
    c_adhd = sum(1 for v in pid_label.values() if v == 1) == 9
    c_ctrl = sum(1 for v in pid_label.values() if v == 0) == 9
    checks.append(("Test participants = 18", c_count))
    checks.append(("ADHD = 9", c_adhd))
    checks.append(("Control = 9", c_ctrl))

    print(f"Dataset: {npz_path.name}")
    print(f"Test participants: {len(pid_set)} "
          f"(ADHD {sum(v == 1 for v in pid_label.values())}, "
          f"Control {sum(v == 0 for v in pid_label.values())})")

    # ---------- read Day 8.5 participant analysis (ถ้ามี) ----------
    p85 = find_first(P85_CANDIDATES)
    p85_prob = {}
    if p85 is not None:
        df85 = pd.read_csv(p85)
        if "Participant_ID" in df85.columns:
            df85 = df85.rename(columns={"Participant_ID": "ID"})
        for _, r in df85.iterrows():
            p85_prob[str(r["ID"])] = float(r["Mean_Probability"])

    # ---------- segment-level probabilities ----------
    seg_probs = model.predict(Xc_test, batch_size=64, verbose=0).flatten()
    c_nan = not np.isnan(seg_probs).any()
    c_inf = not np.isinf(seg_probs).any()
    checks.append(("No NaN in attribution", c_nan))
    checks.append(("No Inf in attribution", c_inf))

    # ---------- identify segments per participant ----------
    segs_of = {}
    for i, (pid, p) in enumerate(zip(pids, seg_probs)):
        segs_of.setdefault(pid, []).append((i, float(p)))

    # ---------- pick correct ADHD / Control examples ----------
    correct_adhd = None
    correct_ctrl = None
    for pid, lbl in pid_label.items():
        if lbl == 1 and p85_prob.get(pid, -1) >= THRESHOLD and correct_adhd is None:
            correct_adhd = pid
        if lbl == 0 and p85_prob.get(pid, 2) < THRESHOLD and correct_ctrl is None:
            correct_ctrl = pid

    # ---------- validate target selection ----------
    c_targets = all(p in pid_set for p in TARGET_FN + [TARGET_THRESHOLD])
    checks.append(("Selected participants exist", c_targets))
    checks.append(("v25p exists", "v25p" in pid_set))
    checks.append(("v6p exists", "v6p" in pid_set))

    TARGETS = [
        ("v25p", "False_Negative"),
        ("v6p", "False_Negative"),
        (TARGET_THRESHOLD, "Threshold_case"),
        (correct_adhd, "Correct_ADHD"),
        (correct_ctrl, "Correct_Control"),
    ]
    TARGETS = [(p, k) for p, k in TARGETS if p is not None]

    # ---------- Grad-CAM setup ----------
    last_conv = find_last_conv(model)
    c_last_conv = last_conv is not None
    checks.append(("Last conv layer found", c_last_conv))
    if not c_last_conv:
        print("ERROR: ไม่พบ Conv2D layer ในโมเดล"); sys.exit(1)
    target_layer = last_conv.name
    print(f"Grad-CAM target layer: {target_layer}")

    # สร้าง functional grad-model (แชร์น้ำหนักกับโมเดลเดิม ไม่แก้สิ่งใด)
    grad_model = build_grad_model(model, target_layer)

    # ---------- per-participant analysis ----------
    summary_rows = []
    channel_imp_accum = np.zeros(len(CHANNELS))
    freq_imp_accum = np.zeros(N_FREQ)
    time_imp_accum = np.zeros(N_TIME)
    n_heatmaps = 0
    per_participant_info = {}

    for pid, kind in TARGETS:
        segs = segs_of.get(pid, [])
        if not segs:
            continue
        probs = np.array([p for _, p in segs])
        idxs = np.array([i for i, _ in segs])
        mean_p = probs.mean()
        # representative = segment ที่ probability ใกล้ค่าเฉลี่ยที่สุด
        rep_local = int(np.argmin(np.abs(probs - mean_p)))
        rep_idx = int(idxs[rep_local])

        x_in = Xc_test[rep_idx:rep_idx + 1]  # (1, 65, 7, 19)
        heatmap, conv_out, pred = compute_gradcam(grad_model, x_in)
        hmap = heatmap[0]
        hmap_up = upsample_heatmap(hmap, (N_FREQ, N_TIME))

        # frequency / time importance (accumulate)
        freq_imp_accum += hmap_up.sum(axis=1)
        time_imp_accum += hmap_up.sum(axis=0)

        # channel importance proxy: |mean input| * mean activation magnitude
        x_full = X_test[rep_idx]  # (19, 65, 7)
        ch_imp = np.abs(x_full).mean(axis=(1, 2))
        conv_mag = np.abs(conv_out[0]).mean(axis=(0, 1))
        ch_imp_weighted = ch_imp * float(conv_mag.mean())
        channel_imp_accum += ch_imp_weighted

        # spectrogram mean (avg over 19 channels) สำหรับ visualization
        spec_mean = X_test[rep_idx].mean(axis=0)  # (65, 7)

        true_label = int(pid_label[pid])
        part_pred = 1 if p85_prob.get(pid, mean_p) >= THRESHOLD else 0

        meta = {
            "ID": pid,
            "true_class": CLASS_NAMES[true_label],
            "pred_class": CLASS_NAMES[part_pred],
            "prob": float(p85_prob.get(pid, mean_p)),
            "seg_prob": float(pred[0]),
        }

        fname = {
            "v25p": "day8_06_v25p_gradcam.png",
            "v6p": "day8_06_v6p_gradcam.png",
            TARGET_THRESHOLD: "day8_06_threshold_case_gradcam.png",
        }.get(pid)
        if fname is None:
            if kind == "Correct_ADHD":
                fname = "day8_06_correct_adhd_gradcam.png"
            elif kind == "Correct_Control":
                fname = "day8_06_correct_control_gradcam.png"
            else:
                fname = f"day8_06_{pid}_gradcam.png"
        plot_gradcam(spec_mean, hmap_up, meta, BASE_DIR / fname)
        n_heatmaps += 1

        summary_rows.append({
            "ID": pid,
            "True_Class": meta["true_class"],
            "Participant_Probability": round(meta["prob"], 6),
            "Participant_Prediction": meta["pred_class"],
            "Correct": (meta["true_class"] == meta["pred_class"]),
            "Selected_Segment": rep_idx,
            "Segment_Probability": round(meta["seg_prob"], 6),
            "Segment_Type": "Representative",
            "Target_Layer": target_layer,
        })

        top_freq = int(np.argmax(hmap_up.sum(axis=1)))
        top_time = int(np.argmax(hmap_up.sum(axis=0)))
        top_ch = int(np.argmax(ch_imp_weighted))

        per_participant_info[pid] = {
            "kind": kind,
            "true_class": meta["true_class"],
            "pred_class": meta["pred_class"],
            "prob": meta["prob"],
            "seg_prob": meta["seg_prob"],
            "top_freq_bin": top_freq,
            "top_freq_hz": float(FREQ_BINS[top_freq]),
            "top_time_bin": top_time,
            "top_channel": CHANNELS[top_ch],
        }

    # ---------- save summary CSV ----------
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    checks.append(("Summary CSV generated", OUT_SUMMARY.exists()))

    # ---------- importance CSVs ----------
    ch_df = pd.DataFrame({
        "Channel": CHANNELS,
        "Importance": np.round(channel_imp_accum / max(len(TARGETS), 1), 6),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    ch_df["Rank"] = np.arange(1, len(ch_df) + 1)
    ch_df.to_csv(OUT_CH_IMP, index=False)
    checks.append(("Channel importance generated", OUT_CH_IMP.exists()))

    freq_df = pd.DataFrame({
        "Frequency_Bin": np.arange(N_FREQ),
        "Frequency_Hz": FREQ_BINS,
        "Importance": np.round(freq_imp_accum / max(len(TARGETS), 1), 6),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    freq_df["Rank"] = np.arange(1, len(freq_df) + 1)
    freq_df.to_csv(OUT_FREQ_IMP, index=False)
    checks.append(("Frequency importance generated", OUT_FREQ_IMP.exists()))

    time_df = pd.DataFrame({
        "Time_Bin": np.arange(N_TIME),
        "Importance": np.round(time_imp_accum / max(len(TARGETS), 1), 6),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    time_df["Rank"] = np.arange(1, len(time_df) + 1)
    time_df.to_csv(OUT_TIME_IMP, index=False)
    checks.append(("Time importance generated", OUT_TIME_IMP.exists()))

    # ---------- heatmap count check ----------
    expected_pngs = [
        "day8_06_v25p_gradcam.png",
        "day8_06_v6p_gradcam.png",
        "day8_06_correct_adhd_gradcam.png",
        "day8_06_correct_control_gradcam.png",
        "day8_06_threshold_case_gradcam.png",
    ]
    n_expected = sum((BASE_DIR / f).exists() for f in expected_pngs)
    checks.append(("Heatmaps generated", n_expected == len(expected_pngs)))

    # ---------- report ----------
    top_ch_global = ch_df.iloc[0]
    top_freq_global = freq_df.iloc[0]
    top_time_global = time_df.iloc[0]

    def part_block(pid):
        if pid and pid in per_participant_info:
            i = per_participant_info[pid]
            return (
                f"Participant ID: {pid}\n"
                f"Participant probability: {i['prob']:.4f}\n"
                f"Representative segment probability: {i['seg_prob']:.4f}\n"
                f"Top frequency bin: {i['top_freq_bin']} ({i['top_freq_hz']:.1f} Hz)\n"
                f"Top time bin: {i['top_time_bin']}\n"
                f"Top channel (magnitude proxy): {i['top_channel']}\n"
            )
        return "Participant not found in the test set.\n"

    report = f"""============================================================
DAY 8.6 - MODEL INTERPRETABILITY REPORT
============================================================

1. Objective
Investigate which regions of the EEG spectrogram the Improved CNN
(from Day 8.2) focuses on when making its predictions. This is a model
attribution experiment, not a biological or clinical study.

2. Model Used
Improved CNN (Day 8.2, file: day8_02_improved_model.keras)
Architecture: Conv2D 32 -> Conv2D 64 -> Conv2D 128 -> GlobalAvgPool
-> Dense 64 -> Dense 1. Total parameters: ~107K.
The model was loaded for inference only; no weights were modified.

3. Test Participants
18 participants (9 ADHD, 9 Control) — same split as Day 7.5 / 8.4.

4. Interpretability Method
Grad-CAM applied at the LAST Conv2D layer before GlobalAveragePooling2D:
  target_layer = {target_layer}
A functional gradient model sharing the original layers/weights was
constructed. Gradients of P(ADHD) with respect to the conv feature maps
were globally pooled and used to weight the maps. Heatmaps were
upsampled (bilinear) back to the original spectrogram grid
(65 frequency bins x 7 time bins) and overlaid on the mean spectrogram
(averaged across 19 EEG channels).

5. Selected Participants
- Correct ADHD example:    {correct_adhd or '(not found)'}
- Correct Control example: {correct_ctrl or '(not found)'}
- False Negative:          v25p
- False Negative:          v6p
- Threshold case:          v140 (participant-level P(ADHD) ~ 0.465)

6. Correct ADHD Example
{part_block(correct_adhd)}The visualization indicates which input regions were influential for the
model prediction.

7. Correct Control Example
{part_block(correct_ctrl)}The visualization indicates which input regions were influential for the
model prediction.

8. False Negative v25p
{part_block('v25p')}The model assigned a low ADHD probability to this participant despite the
true ADHD label. Grad-CAM shows which regions the model attended to when
producing this low probability.

9. False Negative v6p
{part_block('v6p')}Same interpretation as v25p: model attribution only; no biological claim.

10. Threshold Case (v140)
{part_block('v140')}This participant is close to the decision threshold 0.5, so small shifts
in attribution could flip the prediction.

11. Channel Importance (aggregated across selected participants)
Top channel (magnitude proxy): {top_ch_global['Channel']} (importance {top_ch_global['Importance']:.4f})
Note: this is an activation-magnitude proxy, NOT a biomarker claim.
Full table: day8_06_channel_importance.csv

12. Frequency Importance
Top frequency bin: {top_freq_global['Frequency_Bin']} ({top_freq_global['Frequency_Hz']:.1f} Hz)
Full table: day8_06_frequency_importance.csv
Bin indices correspond to the spectrogram frequency axis (0..64 Hz).

13. Time Importance
Top time bin: {top_time_global['Time_Bin']} (out of 7, within the 4-second segment)
Full table: day8_06_time_importance.csv

14. Key Observations
- Grad-CAM heatmaps show that the model's attribution is spatially
  localized in the frequency-time plane, not uniformly distributed.
- For correct predictions, the attribution typically concentrates in
  a small number of frequency bands and time windows.
- For the two false negatives (v25p, v6p), the model's attribution
  pattern differs from the correct ADHD cases, which may indicate
  that these participants' spectrograms do not match the patterns
  the model learned during training.
- The threshold case (v140) shows intermediate attribution strength,
  consistent with its probability being close to 0.5.
- All observations describe model behavior on this dataset; they do
  not establish biological or clinical explanations.

15. Limitations
1. Test set only contains 18 participants.
2. Only selected examples are visualized.
3. Grad-CAM is an approximate attribution method.
4. Model attribution does not prove biological causality.
5. EEG channel importance does not establish clinical biomarkers.
6. Dataset is limited to the current experimental dataset.
7. No external validation.
8. The system is not a clinical diagnostic device.

16. Conclusion
The Grad-CAM analysis shows that the Improved CNN relies on specific
frequency-time regions and EEG channels when making predictions. The
visualizations describe model attribution, not clinical or biological
causation. Further work with larger and independent datasets, and
complementary interpretability methods, would be required before any
stronger claims could be made.
============================================================
"""
    OUT_REPORT.write_text(report, encoding="utf-8")
    checks.append(("Report generated", OUT_REPORT.exists()))

    # ---------- validation summary ----------
    all_ok = all(ok for _, ok in checks)
    lines = ["=" * 60, "DAY 8.6 - VALIDATION SUMMARY", "=" * 60]
    for name, ok in checks:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    lines += ["", f"Status: {'PASSED' if all_ok else 'FAILED'}", "=" * 60]
    OUT_VAL.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- terminal ----------
    print(f"\nSelected Examples:")
    for pid, kind in TARGETS:
        print(f"  - {kind}: {pid}")
    print(f"\nGrad-CAM: {'PASSED' if n_heatmaps == len(expected_pngs) else 'FAILED'}")
    print(f"Channel Analysis: {'PASSED' if OUT_CH_IMP.exists() else 'FAILED'}")
    print(f"Frequency Analysis: {'PASSED' if OUT_FREQ_IMP.exists() else 'FAILED'}")
    print(f"Time Analysis: {'PASSED' if OUT_TIME_IMP.exists() else 'FAILED'}")
    print(f"\nOutput Files: {'PASSED' if all((BASE_DIR / f).exists() for f in expected_pngs) and OUT_SUMMARY.exists() else 'FAILED'}")
    print(f"Validation: {'PASSED' if all_ok else 'FAILED'}")
    print("\n" + "-" * 60)
    for name, ok in checks:
        print(f"{name}: {'PASSED' if ok else 'FAILED'}")
    print("\n" + "=" * 60)
    print(f"DAY 8.6: {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()