# ============================================================
# DAY 10.1 - PRODUCT INTEGRATION AUDIT
# NeuroScreen AI (Research Prototype / Screening Support)
# ตรวจสอบโครงสร้างความพร้อมก่อนเชื่อม Improved CNN กับ Web App
# - อ่านอย่างเดียว ไม่แก้ไขไฟล์เดิม ไม่เทรน ไม่สร้างแอป
# - ไม่โหลด TensorFlow / ไม่โหลด adhdata.csv
# ============================================================

import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent   # .../day10_Product_Development/day10
DATA_DIR = BASE_DIR.parent.parent            # .../Data
PROJECT_ROOT = DATA_DIR.parent               # .../NeuroScreen-AI

OUT_REPORT = BASE_DIR / "day10_01_integration_audit.txt"

# ---------------- expected spec (จากผลการทดลองเดิม) ----------------
EXPECTED = {
    "channels": 19,
    "sampling_rate": 128,
    "window_sec": 4.0,
    "window_samples": 512,
    "hop_samples": 64,
    "frames": 7,
    "nperseg": 128,
    "freq_bins": 65,
    "model_input_shape": [None, 65, 7, 19],
    "threshold": 0.5,
}

PATTERNS = {
    "sampling_rate": r"\b(?:FS|SAMPLING_RATE|FS_HZ|fs)\s*=\s*(\d+)",
    "nperseg": r"\bnperseg\s*=\s*(\d+)",
    "noverlap": r"\bnoverlap\s*=\s*(\d+)",
    "hop_samples": r"\b(?:HOP_SAMPLES|HOP|hop)\s*=\s*(\d+)",
    "window_sec": r"\b(?:WINDOW_SEC|WINDOW_SECONDS|SEGMENT_SECONDS|window_sec)\s*=\s*([\d.]+)",
    "window_samples": r"\b(?:WINDOW_SAMPLES|SEGMENT_SAMPLES|window_samples)\s*=\s*(\d+)",
    "frames": r"\b(?:N_FRAMES|FRAMES|n_frames)\s*=\s*(\d+)",
    "freq_bins": r"\b(?:N_FREQ|FREQ_BINS|n_freq)\s*=\s*(\d+)",
    "channels": r"\b(?:N_CHANNELS|NUM_CHANNELS|n_channels)\s*=\s*(\d+)",
    "threshold": r"\b(?:THRESHOLD|threshold)\s*=\s*(0\.\d+)",
}


def find_files(pattern):
    return sorted(DATA_DIR.rglob(pattern))


def find_one(pattern):
    m = find_files(pattern)
    return m[0] if m else None


def fmt_path(p):
    return str(p) if p else "NOT FOUND"


def fmt_size(p):
    try:
        return f"{p.stat().st_size / 1024:.1f} KB" if p.stat().st_size < 1024 * 1024 \
            else f"{p.stat().st_size / (1024 * 1024):.1f} MB"
    except Exception:
        return "?"


def extract_constants(path):
    """ดึงค่า config จากโค้ดเดิมด้วย regex (อ่านอย่างเดียว)"""
    if path is None or not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m:
            found[key] = m.group(1)
    # ตรวจการรวมผลระดับ participant (mean probability)
    if re.search(r"Mean_Probability|mean probability|\.mean\(\)", text, re.I):
        found["participant_aggregation"] = "mean segment probability"
    return found


def peek_keras_model(path):
    """อ่าน input shape / layers จาก .keras (ZIP) โดยไม่ต้องใช้ TensorFlow"""
    info = {"input_shape": None, "layers": [], "error": None}
    try:
        with zipfile.ZipFile(path) as z:
            if "config.json" in z.namelist():
                cfg = json.loads(z.read("config.json"))
                layers = cfg.get("config", {}).get("layers", [])
                for l in layers:
                    info["layers"].append(l.get("class_name"))
                    lc = l.get("config", {})
                    if "batch_input_shape" in lc and info["input_shape"] is None:
                        info["input_shape"] = lc["batch_input_shape"]
    except Exception as e:
        info["error"] = str(e)
    return info


def inspect_npz(path):
    try:
        with np.load(path) as data:
            return {k: tuple(data[k].shape) for k in data.files}, None
    except Exception as e:
        return None, str(e)


def main():
    print("=" * 60)
    print("DAY 10.1 - PRODUCT INTEGRATION AUDIT")
    print("=" * 60)

    # ---------- 1. locate files ----------
    model = find_one("day8_02_improved_model.keras") or find_one("*improved*.keras")
    npz = find_one("day7_02_cnn_dataset.npz")
    adhdata = find_one("adhdata.csv")

    code_82 = find_one("day8_02_improved_cnn.py") or find_one("day8_02*.py")
    code_75 = find_one("day7_05*.py")
    code_85 = find_one("day8_05*.py")
    code_72 = find_one("day7_02*.py")
    code_73 = find_one("day7_03*.py")

    codes_5 = find_files("day5*.py")
    codes_6 = find_files("day6*.py")
    code_41 = find_one("day4_01*.py")

    app_files = sorted(PROJECT_ROOT.glob("*.py")) + sorted(DATA_DIR.glob("*.py"))
    req_files = sorted(PROJECT_ROOT.glob("requirements*.txt")) + \
                sorted(DATA_DIR.glob("requirements*.txt"))

    # ---------- 2. inspect artifacts ----------
    model_info = peek_keras_model(model) if model else {"input_shape": None, "error": "model not found"}
    npz_shapes, npz_err = inspect_npz(npz) if npz else (None, "npz not found")

    # ---------- 3. extract constants from original code ----------
    detected, sources = {}, {}
    scan_list = [code_41] + codes_5 + codes_6 + [code_72, code_73, code_75, code_82, code_85]
    for f in scan_list:
        for k, v in extract_constants(f).items():
            if k not in detected:
                detected[k] = v
                sources[k] = f.name if f else "?"

    # ---------- 4. compatibility checks ----------
    checks = []

    def add_check(item, expected, detected_val, ok):
        checks.append((item, expected, detected_val, ok))

    mis = model_info.get("input_shape")
    add_check("Model input shape", str(EXPECTED["model_input_shape"]),
              str(mis), mis is not None and list(mis)[1:] == [65, 7, 19])

    xtr = npz_shapes.get("X_train") if npz_shapes else None
    add_check("Dataset stored shape (X_train)", "(2992, 19, 65, 7)",
              str(xtr), xtr == (2992, 19, 65, 7))

    for key, exp in [("channels", 19), ("sampling_rate", 128), ("window_sec", 4.0),
                     ("window_samples", 512), ("hop_samples", 64),
                     ("frames", 7), ("nperseg", 128), ("freq_bins", 65),
                     ("threshold", 0.5)]:
        dv = detected.get(key)
        ok = dv is not None and float(dv) == float(exp)
        add_check(key, str(exp), str(dv) if dv else "not detected", ok)

    agg = detected.get("participant_aggregation")
    add_check("Participant aggregation", "mean segment probability",
              agg if agg else "not detected", agg is not None)

    # ---------- 5. build report ----------
    dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()], key=lambda p: p.name)
    tree_lines = []
    for d in dirs:
        files = sorted([f.name for f in d.iterdir() if f.is_file()])
        tree_lines.append(f"- {d.name}/  ({len(files)} files)")
        for fn in files[:6]:
            tree_lines.append(f"    {fn}")
        if len(files) > 6:
            tree_lines.append(f"    ... +{len(files) - 6} more")

    issues = []
    if model is None:
        issues.append("CRITICAL: ไม่พบ Improved CNN model (day8_02_improved_model.keras)")
    if npz is None:
        issues.append("CRITICAL: ไม่พบ dataset day7_02_cnn_dataset.npz")
    if not req_files:
        issues.append("ไม่มี requirements.txt -> ต้องสร้าง (streamlit, tensorflow, numpy, pandas, scipy, scikit-learn, matplotlib, pillow)")
    if not app_files:
        issues.append("ไม่มี app.py -> ต้องสร้าง Streamlit app ใหม่")
    if not codes_5:
        issues.append("ไม่พบโค้ด Day 5 (preprocessing/spectrogram) -> ต้องยืนยันตำแหน่งก่อนสร้างเว็บ")
    if not codes_6:
        issues.append("ไม่พบโค้ด Day 6 (split/segment manifest) -> ต้องยืนยันตำแหน่งก่อนสร้างเว็บ")
    for item, exp, dv, ok in checks:
        if not ok:
            issues.append(f"COMPATIBILITY: {item} คาดหวัง {exp} แต่ตรวจพบ '{dv}'")
    issues.append("MEMORY: โหลดโมเดลครั้งเดียวด้วย @st.cache_resource และจำกัดขนาดไฟล์ EEG ที่อัปโหลด")
    issues.append("RUNTIME: เครื่องนี้ไม่มี GPU (CPU only) -> inference เร็วพอสำหรับ 4s segments แต่ต้องทดสอบเวลาตอบสนอง")

    missing = sum(1 for _, _, _, ok in checks if not ok)

    report = f"""============================================================
DAY 10.1 - PRODUCT INTEGRATION AUDIT REPORT
NeuroScreen AI (research prototype / screening support เท่านั้น
ไม่ใช่ระบบวินิจฉัยทางการแพทย์)
Audit date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
============================================================

1. PROJECT STRUCTURE (ตรวจพบจริง)
Project root: {PROJECT_ROOT}
Data root:    {DATA_DIR}

{chr(10).join(tree_lines)}

Web app files ที่ตรวจพบ:
{chr(10).join(['  - ' + str(p) for p in app_files]) if app_files else '  (ไม่พบ app.py)'}
Requirements files ที่ตรวจพบ:
{chr(10).join(['  - ' + str(p) for p in req_files]) if req_files else '  (ไม่พบ requirements.txt)'}

Dataset ขนาดใหญ่:
- adhdata.csv: {fmt_path(adhdata)} ({fmt_size(adhdata) if adhdata else '-'})
  [ห้ามโหลดทั้งไฟล์ในเว็บ - ใช้เฉพาะเมื่อจำเป็นจริง]

2. MODEL PATH (ตรวจพบ)
- Improved CNN (Final Model Candidate): {fmt_path(model)}
  ขนาด: {fmt_size(model) if model else '-'}
- Training code: {fmt_path(code_82)}

3. PREPROCESSING PATH (ตรวจพบ)
- Day 5 scripts:
{chr(10).join(['    - ' + str(p) for p in codes_5]) if codes_5 else '    NOT FOUND'}
- Day 6 scripts:
{chr(10).join(['    - ' + str(p) for p in codes_6]) if codes_6 else '    NOT FOUND'}
- Day 4.1 window experiment: {fmt_path(code_41)}

4. SPECTROGRAM / DATASET PATH (ตรวจพบ)
- Day 7.2 dataset builder: {fmt_path(code_72)}
- CNN dataset (npz): {fmt_path(npz)}
{('  shapes: ' + ', '.join(f'{k}={v}' for k, v in npz_shapes.items() if k.startswith('X'))) if npz_shapes else '  NOT FOUND'}
- Day 7.3 CNN architecture code: {fmt_path(code_73)}
- Day 7.5 evaluation code: {fmt_path(code_75)}
- Day 8.2 improved training code: {fmt_path(code_82)}
- Day 8.5 error analysis code: {fmt_path(code_85)}

5. MODEL INPUT SHAPE
- Expected: (batch, 65, 7, 19) = freq bins x time bins x EEG channels
- Detected from .keras config: {model_info['input_shape']}
- Stored dataset shape: (N, 19, 65, 7) -> transpose เป็น (N, 65, 7, 19) ก่อนเข้าโมเดล

6. จุดที่เชื่อมต่อกันได้ (PIPELINE COMPATIBILITY)
Pipeline: EEG input -> Preprocessing -> Segmentation -> Spectrogram
          -> Improved CNN -> Participant-level aggregation -> Screening result

รายการตรวจสอบ (คาดหวัง / ตรวจพบ / สถานะ):
"""
    for item, exp, dv, ok in checks:
        report += f"- {item:<28} expected={exp:<24} detected={dv:<24} {'OK' if ok else 'MISMATCH'}\n"

    report += f"""
สรุปความเข้ากันได้: {'ทุกจุดตรงกับสเปกเดิม' if missing == 0 else f'พบ {missing} จุดที่ต้องตรวจสอบ'}
การรวมผลระดับ participant: mean(segment probabilities) แล้วตัดที่ threshold 0.5 (คงที่ ห้ามเปลี่ยนในขั้นตอนนี้)

7. ปัญหาที่ต้องแก้ก่อนสร้างเว็บ
"""
    for i, s in enumerate(issues, 1):
        report += f"{i}. {s}\n"

    report += """
8. แผนพัฒนา WEB APPLICATION (Streamlit)
Framework: Streamlit (ตัวเลือกเริ่มต้น เนื่องจากโปรเจกต์ยังไม่มี framework อื่น)
โครงสร้างหน้าเว็บ:
  1. Home - แนะนำ NeuroScreen AI พร้อมคำเตือนว่านี่คือ research prototype
     สำหรับ screening support เท่านั้น ไม่ใช่เครื่องมือวินิจฉัยทางการแพทย์
  2. EEG Input - อธิบายรูปแบบข้อมูลที่รองรับ: 19 EEG channels, 128 Hz,
     ไฟล์ .csv/.txt ที่ระบุ channel และเวลา; แนะนำความยาวขั้นต่ำ 4 วินาที
     ต่อ recording; จำกัดขนาดไฟล์เพื่อคุมหน่วยความจำ
  3. Screening Analysis - รัน pipeline เดียวกับการเทรน:
     EEG -> bandpass/notch -> standardize -> 4s segments (hop 64 samples)
     -> STFT (nperseg=128) -> (65, 7, 19) -> Improved CNN -> segment probs
  4. Results - แสดง participant-level mean probability, predicted class
     ที่ threshold 0.5, segment-level distribution และคำอธิบายที่ระมัดระวัง
     (ห้ามสร้างผลปลอมหรือข้อมูลตัวอย่างเสมือนเป็นผลจากโมเดลจริง)
  5. About & Limitations - ข้อจำกัดทั้งหมด: test set 18 คน, ไม่มี external
     validation, fixed threshold 0.5, ผลเป็น experimental เท่านั้น

เทคนิค:
- โหลดโมเดลครั้งเดียวด้วย @st.cache_resource
- cache ผลลัพธ์การประมวลผลต่อไฟล์ด้วย @st.cache_data (hash input)
- แสดงคำเตือน (disclaimer) ทุกหน้าที่เกี่ยวข้องกับผลทำนาย
- ไม่โหลด adhdata.csv ในเว็บ

9. รายการไฟล์ที่ต้องสร้างหรือแก้ไขในขั้นถัดไป (Day 10.2+)
- data/day10_Product_Development/day10/day10_02_inference_pipeline.py
  (โมดูลรวม: โหลดโมเดล + ทำความสะอาด + แบ่งเซกเมนต์ + spectrogram
   + ทำนาย + รวมผลระดับผู้เข้าร่วม — คัดลอกจากโค้ด Day 5/7/8.2 โดยตรง
   ห้ามแก้ไขไฟล์เดิม)
- data/day10_Product_Development/day10/app.py (Streamlit UI 5 หน้า)
- data/day10_Product_Development/day10/requirements.txt
- ไฟล์ทดสอบขนาดเล็ก (เช่น สคริปต์สร้าง synthetic EEG สำหรับทดสอบ
  pipeline โดยไม่แตะข้อมูลจริง)

หมายเหตุ: ห้ามเทรนโมเดลใหม่ ห้ามแก้ไฟล์ Day 7-9 ห้ามเปลี่ยน threshold 0.5
และห้ามอ้างว่าระบบวินิจฉัยโรคได้
============================================================
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    # ---------- terminal ----------
    print(f"\nReport saved: {OUT_REPORT}\n")
    print("Located files:")
    print(f"  Improved CNN model : {fmt_path(model)}")
    print(f"  CNN dataset (npz)  : {fmt_path(npz)}")
    print(f"  Day 5 scripts      : {len(codes_5)} found")
    print(f"  Day 6 scripts      : {len(codes_6)} found")
    print(f"  Day 7.2 builder    : {fmt_path(code_72)}")
    print(f"  Day 7.5 evaluation : {fmt_path(code_75)}")
    print(f"  Day 8.2 training   : {fmt_path(code_82)}")
    print(f"  Day 8.5 analysis   : {fmt_path(code_85)}")
    print(f"  app.py candidates  : {len(app_files)}")
    print(f"  requirements.txt   : {len(req_files)}")

    print("\nCompatibility checks:")
    for item, exp, dv, ok in checks:
        print(f"  {item:<28} expected={exp:<22} detected={dv:<22} {'OK' if ok else 'MISMATCH'}")

    print("\nIssues:")
    for s in issues:
        print(f"  - {s}")

    print("\n" + "=" * 60)
    if model and npz:
        print("DAY 10.1 AUDIT: COMPLETED")
    else:
        print("DAY 10.1 AUDIT: COMPLETED WITH CRITICAL MISSING FILES")
    print("=" * 60)


if __name__ == "__main__":
    main()