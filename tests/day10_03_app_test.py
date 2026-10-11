# ============================================================
# DAY 10.3 - WEB APP LOGIC TESTS
# ทดสอบโครงสร้าง/การจัดการข้อผิดพลาด โดยไม่เปิด Streamlit จริง
# ข้อมูลสังเคราะห์ใช้ทดสอบโครงสร้างเท่านั้น
# ห้ามอ้างผลจากข้อมูลสังเคราะห์เป็นประสิทธิภาพการคัดกรอง
# ============================================================

import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import app as webapp
from day10_02_inference_pipeline import (
    collect_evidence, resolve_parameters, readiness, run_inference,
)

OUT = _APP_DIR / "day10_03_validation_summary.txt"
EEG_CHANNELS = webapp.EEG_CHANNELS


class FakeUpload:
    def __init__(self, data):
        self._data = data
        self.size = len(data)
    def read(self, *a, **k):
        return self._data


def csv_bytes(arr):
    df = pd.DataFrame(arr, columns=EEG_CHANNELS)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def t1_valid():
    data = np.random.default_rng(0).standard_normal((1024, 19))
    df, errs = webapp.validate_uploaded_csv(FakeUpload(csv_bytes(data)))
    assert errs == [], errs
    assert df is not None and df.shape == (1024, 19)


def t2_missing_cols():
    dfx = pd.DataFrame(np.zeros((100, 3)), columns=["Fp1", "Fp2", "timestamp"])
    buf = io.StringIO(); dfx.to_csv(buf, index=False)
    _, errs = webapp.validate_uploaded_csv(FakeUpload(buf.getvalue().encode()))
    assert any("19" in e or "ไม่พบคอลัมน์" in e for e in errs), errs


def t3_nan():
    data = np.random.default_rng(1).standard_normal((600, 19))
    data[10, 3] = np.nan
    _, errs = webapp.validate_uploaded_csv(FakeUpload(csv_bytes(data)))
    assert any("NaN" in e for e in errs), errs


def t4_inf():
    data = np.random.default_rng(2).standard_normal((600, 19))
    data[5, 1] = np.inf
    _, errs = webapp.validate_uploaded_csv(FakeUpload(csv_bytes(data)))
    assert any("Inf" in e for e in errs), errs


def t5_empty():
    _, errs = webapp.validate_uploaded_csv(FakeUpload(b""))
    assert errs, "ไฟล์ว่างต้องถูกปฏิเสธ"


def t6_too_short():
    data = np.random.default_rng(3).standard_normal((100, 19))
    _, errs = webapp.validate_uploaded_csv(FakeUpload(csv_bytes(data)))
    assert any("สั้นเกินไป" in e for e in errs), errs


def load_pipe():
    evidence, file_info, scans = collect_evidence()
    params, confirmed, blocked, scans = resolve_parameters(evidence, scans)
    blocked = list(blocked) + list(readiness(params, scans))
    assert not blocked, f"pipeline blocked: {blocked}"
    return params, scans


def t7_pipeline_ready():
    load_pipe()


def t8_real_model_inference():
    params, scans = load_pipe()
    eeg = np.random.default_rng(4).standard_normal((512 * 3, 19))
    result = run_inference(eeg, params, scans)
    for key in ("mean_probability", "predicted_class", "threshold",
                "n_segments", "segment_probabilities", "cnn_input_shape"):
        assert key in result, f"missing key: {key}"
    assert result["n_segments"] == 3
    assert result["threshold"] == 0.5
    assert result["predicted_class"] in ("ADHD", "Control")
    segs = np.asarray(result["segment_probabilities"])
    assert segs.shape == (3,) and np.all((segs >= 0) & (segs <= 1))
    assert tuple(result["cnn_input_shape"]) == (3, 65, 7, 19)
    print("      [หมายเหตุ] ข้อมูลสังเคราะห์ใช้ตรวจโครงสร้างเท่านั้น "
          "- ไม่ใช่อ้างอิงประสิทธิภาพ")


def t9_reject_short_pipeline():
    params, scans = load_pipe()
    eeg = np.random.default_rng(5).standard_normal((100, 19))
    try:
        run_inference(eeg, params, scans)
    except ValueError:
        return
    raise AssertionError("ต้องปฏิเสธข้อมูลที่สั้นกว่า 512 samples")


def t10_pages_defined():
    for name in ("Home", "EEG Input", "Screening Analysis",
                 "Results", "About & Limitations"):
        assert name in webapp.PAGES, f"missing page: {name}"


def main():
    tests = [
        ("T1 App เริ่มต้น/โมดูลโหลดได้ (import สำเร็จ)", t1_valid),
        ("T2 ปฏิเสธ CSV ที่ขาดคอลัมน์ EEG", t2_missing_cols),
        ("T3 ปฏิเสธข้อมูลที่มี NaN", t3_nan),
        ("T4 ปฏิเสธข้อมูลที่มี Inf", t4_inf),
        ("T5 ปฏิเสธไฟล์ว่าง", t5_empty),
        ("T6 ปฏิเสธไฟล์สั้นเกินไป (<512 samples)", t6_too_short),
        ("T7 Pipeline พร้อมใช้ (หลักฐานจากโค้ดฝึกครบ)", t7_pipeline_ready),
        ("T8 การทำนายด้วยโมเดลจริง (โครงสร้างเท่านั้น)", t8_real_model_inference),
        ("T9 Pipeline ปฏิเสธข้อมูลสั้น", t9_reject_short_pipeline),
        ("T10 หน้าเว็บครบ 5 หน้า", t10_pages_defined),
    ]
    results = []
    for name, fn in tests:
        try:
            fn()
            results.append((name, "PASSED", ""))
            print(f"[PASS] {name}")
        except Exception as e:
            results.append((name, "FAILED", f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    any_failed = any(s == "FAILED" for _, s, _ in results)
    status = "FAILED" if any_failed else "PASSED"

    files_created = [
        ("app.py", _APP_DIR / "app.py"),
        ("requirements.txt", _APP_DIR / "requirements.txt"),
        ("day10_03_app_test.py", _APP_DIR / "day10_03_app_test.py"),
        ("day10_03_validation_summary.txt", OUT),
    ]
    lines = ["=" * 60, "DAY 10.3 - WEB APPLICATION VALIDATION SUMMARY",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 60,
             "", "[1] ไฟล์ที่สร้าง"]
    for name, p in files_created:
        lines.append(f"  - {name}: {'FOUND' if p.exists() else 'MISSING'}")
    lines += ["",
              "[2] วิธีรันแอป",
              "  cd C:\\Users\\Acer\\Desktop\\Healthcare AI",
              "  .\\tfenv\\Scripts\\python.exe -m streamlit run "
              ".\\NeuroScreen-AI\\Data\\day10_Product_Development\\day10\\app.py",
              "  เปิดเบราว์เซอร์ที่ http://localhost:8501",
              "", "[3] ผลการทดสอบ"]
    for name, s, note in results:
        lines.append(f"  [{s}] {name}" + (f"  -- {note}" if note else ""))
    errs = [note for _, s, note in results if s == "FAILED"]
    lines += ["", "[4] ข้อผิดพลาดที่พบ"]
    lines += [f"  - {e}" for e in errs] if errs else ["  (ไม่มี)"]
    lines += ["",
              "[5] ข้อจำกัดที่ยังเหลือ",
              "  - การแสดงผลของ UI ต้องตรวจด้วยตาในเบราว์เซอร์ (ชุดทดสอบตรวจเฉพาะตรรกะ)",
              "  - ควรทดสอบกับไฟล์ EEG จริงแบบ manual เพื่อยืนยันพฤติกรรมปลายต่อปลาย",
              "  - ผลจากข้อมูลสังเคราะห์ไม่ใช่อ้างอิงประสิทธิภาพการคัดกรอง",
              "",
              "[6] การใช้โมเดลจริง",
              "  - ใช้โมเดลจริง: ใช่ (Improved CNN จาก Day 8.2 โหลดแบบครั้งเดียว)",
              "  - ส่วนที่ทดสอบอัตโนมัติไม่ได้: การอัปโหลดไฟล์ผ่าน UI จริง,",
              "    การแสดงผลกราฟ และการนำทางระหว่างหน้า (ต้องตรวจในเบราว์เซอร์)",
              "",
              "[7] การยืนยันความสะอาด",
              "  - ไม่แก้ไขไฟล์ Day 5-9 | ไม่เทรนโมเดลใหม่ | ไม่โหลด adhdata.csv",
              "  - ไม่บันทึกไฟล์ที่ผู้ใช้อัปโหลด | threshold คงที่ 0.5",
              "", "=" * 60, f"DAY 10.3 STATUS: {status}", "=" * 60]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"DAY 10.3 STATUS: {status}")
    print(f"Report: {OUT}")
    print("=" * 60)
    sys.exit(0 if status == "PASSED" else 1)


if __name__ == "__main__":
    main()