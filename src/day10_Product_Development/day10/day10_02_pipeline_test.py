# ============================================================
# DAY 10.2 - PIPELINE TEST RUNNER (REV.2)
# ข้อมูลสังเคราะห์ใช้ทดสอบ "โครงสร้าง" เท่านั้น
# ห้ามอ้างอิงผลจากข้อมูลสังเคราะห์เป็นประสิทธิภาพการคัดกรอง
# ============================================================

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import day10_02_inference_pipeline as P

OUT_SUMMARY = BASE_DIR / "day10_02_validation_summary.txt"
RNG = np.random.default_rng(42)


def mtimes(paths):
    return {str(p): p.stat().st_mtime for p in paths if p.exists()}


def main():
    print("=" * 60)
    print("DAY 10.2 - PIPELINE TEST")
    print("=" * 60)

    watched = list(P.REFERENCE_FILES.values()) + [P.MODEL_PATH]
    before = mtimes(watched)

    # ---------- 1. เก็บหลักฐานจากโค้ดต้นฉบับ ----------
    evidence, file_info, scans = P.collect_evidence()
    params, confirmed, blocked, scans = P.resolve_parameters(evidence, scans)
    blocked += P.readiness(params, scans)

    print("\nReference files:")
    for name, info in file_info.items():
        print(f"  {name}: {'FOUND' if info.get('exists') else 'MISSING'}")

    print("\nCONFIRMED parameters:")
    for c in confirmed:
        print(f"  [CONFIRMED] {c}")
    if blocked:
        print("\nBLOCKED items:")
        for b in dict.fromkeys(blocked):
            print(f"  [BLOCKED] {b}")

    # ---------- 2. ตรวจ input shape จากโมเดลจริง ----------
    model_shape, shape_err = P.peek_model_input_shape()
    shape_match = (model_shape is not None
                   and list(model_shape)[1:] == [params.get("n_freq"),
                                                 params.get("n_time"),
                                                 params.get("n_channels")])
    print(f"\nModel input shape (จากไฟล์ .keras จริง): {model_shape}")
    if shape_err:
        print(f"  ERROR: {shape_err}")
        blocked.append(f"ตรวจ input shape ของโมเดลไม่ได้: {shape_err}")

    results = []

    def run_test(name, fn):
        try:
            fn()
            results.append((name, "PASSED", ""))
            print(f"  [PASS] {name}")
        except P.PipelineBlockedError as e:
            results.append((name, "BLOCKED", "; ".join(e.blocked_items)))
            print(f"  [BLOCKED] {name}: {e}")
        except AssertionError as e:
            results.append((name, "FAILED", str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            results.append((name, "FAILED", f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")

    print("\nTests:")

    # T1: รูปร่างข้อมูลก่อนเข้าโมเดล
    def t1_shape():
        if blocked:
            raise P.PipelineBlockedError(blocked)
        win = params["window_samples"]
        x = RNG.standard_normal((win * 4, params["n_channels"]))
        xv = P.validate_eeg_input(x, params)
        segs = P.segment_eeg(xv, win)
        assert segs.shape == (4, win, params["n_channels"]), f"segment shape ผิด: {segs.shape}"
        prepared = np.stack([P.preprocess_segment(s, params) for s in segs])
        spec = P.make_spectrograms(prepared, params)
        assert spec.shape == (4, params["n_channels"], params["n_freq"], params["n_time"]), \
            f"spectrogram shape ผิด: {spec.shape}"
        X = P.to_cnn_input(spec)
        assert X.shape == (4, params["n_freq"], params["n_time"], params["n_channels"]), \
            f"CNN input shape ผิด: {X.shape}"
        assert np.isfinite(X).all(), "พบ NaN/Inf หลังสร้าง spectrogram"
        # ส่วนที่ท้ายไม่ถึง 512 ต้องถูกตัดทิ้ง (ตรงกับโค้ดฝึก)
        x2 = RNG.standard_normal((win * 2 + 100, params["n_channels"]))
        segs2 = P.segment_eeg(P.validate_eeg_input(x2, params), win)
        assert segs2.shape[0] == 2, "การตัดเศษท้ายไม่ตรงกับโค้ดฝึก"
    run_test("T1 รูปร่างข้อมูลก่อนเข้าโมเดล (N, 65, 7, 19)", t1_shape)

    # T2: การจัดการไฟล์/ข้อมูลที่ไม่ถูกต้อง
    def t2_invalid():
        for bad in [np.zeros((100, 5)), np.zeros(100), np.zeros((0, 19)),
                    np.array([["a", "b"], ["c", "d"]])]:
            try:
                P.validate_eeg_input(bad, params)
                raise AssertionError(f"ควรถูกปฏิเสธ: shape={np.asarray(bad).shape}")
            except ValueError:
                pass
    run_test("T2 การจัดการ input ที่ไม่ถูกต้อง", t2_invalid)

    # T3: NaN / Inf
    def t3_nan_inf():
        if "n_channels" not in params:
            raise P.PipelineBlockedError(["n_channels"])
        n_ch = params["n_channels"]
        a = RNG.standard_normal((600, n_ch)); a[10, 0] = np.nan
        b = RNG.standard_normal((600, n_ch)); b[10, 0] = np.inf
        for bad in (a, b):
            try:
                P.validate_eeg_input(bad, params)
                raise AssertionError("ควรปฏิเสธ NaN/Inf")
            except ValueError:
                pass
    run_test("T3 การปฏิเสธข้อมูลที่มี NaN/Inf", t3_nan_inf)

    # T4: participant-level mean probability (เลขล้วน)
    def t4_aggregate():
        r = P.aggregate_participant([0.2, 0.4, 0.7, 0.8])
        assert abs(r["mean_probability"] - 0.525) < 1e-9, "ค่าเฉลี่ยผิด"
        assert r["predicted_class"] == "ADHD", "0.525 ต้องเป็น ADHD ที่ threshold 0.5"
    run_test("T4 participant-level mean probability", t4_aggregate)

    # T5: threshold 0.5 คงที่ (ขอบเขต)
    def t5_threshold():
        assert P.aggregate_participant([0.5])["predicted_class"] == "ADHD"
        assert P.aggregate_participant([0.49999])["predicted_class"] == "Control"
        assert P.aggregate_participant([0.5])["threshold"] == 0.5
    run_test("T5 threshold 0.5 (boundary, ไม่ปรับเปลี่ยน)", t5_threshold)

    # T6: input shape ของโมเดลตรงกับค่าจากโค้ดต้นฉบับ
    def t6_model_shape():
        assert model_shape is not None, "อ่าน shape จากโมเดลไม่ได้"
        assert shape_match, f"shape โมเดล {model_shape} ไม่ตรงกับค่าที่ยืนยัน"
    run_test("T6 model input shape ตรงกับค่าที่ยืนยัน", t6_model_shape)

    # T7: การทำนายด้วยโมเดลจริง — เฉพาะเมื่อยืนยันครบ
    #     (ข้อมูลสังเคราะห์ => ห้ามอ้างเป็นประสิทธิภาพการคัดกรอง)
    def t7_real_model():
        if blocked:
            raise P.PipelineBlockedError(blocked)
        win = params["window_samples"]
        x = RNG.standard_normal((win * 3, params["n_channels"]))
        res = P.run_inference(x, params, scans)
        model = P.load_model_once()
        assert tuple(model.input_shape[1:]) == (params["n_freq"], params["n_time"],
                                                params["n_channels"]), \
            f"model.input_shape จริงไม่ตรง: {model.input_shape}"
        assert res["segment_probabilities"].shape == (3,), "จำนวน segment ทำนายผิด"
        assert np.all((res["segment_probabilities"] >= 0) & (res["segment_probabilities"] <= 1))
        assert 0.0 <= res["mean_probability"] <= 1.0
        print("      [หมายเหตุ] ใช้ข้อมูลสังเคราะห์เพื่อทดสอบโครงสร้างเท่านั้น "
              "- ไม่ใช่อ้างอิงประสิทธิภาพการคัดกรอง")
    run_test("T7 การทำนายด้วยโมเดลจริง (โครงสร้างเท่านั้น)", t7_real_model)

    # T8: ไฟล์เก่าไม่ถูกแก้ไข
    def t8_no_modification():
        after = mtimes(watched)
        changed = [k for k in before if after.get(k) != before[k]]
        assert not changed, f"ไฟล์เดิมถูกแก้ไข: {changed}"
    run_test("T8 ยืนยันว่าไม่มีการแก้ไขไฟล์เดิม", t8_no_modification)

    # ---------- 3. สรุปสถานะ ----------
    critical_blocked = blocked + [b for n, s, b in results if s == "BLOCKED"]
    any_failed = any(s == "FAILED" for _, s, _ in results)
    if any_failed:
        status = "FAILED"
    elif critical_blocked:
        status = "BLOCKED"
    else:
        status = "PASSED"

    # ---------- 4. รายงาน ----------
    rep = []
    rep.append("=" * 60)
    rep.append("DAY 10.2 - PIPELINE VALIDATION SUMMARY")
    rep.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append("=" * 60)
    rep.append("\n[1] พารามิเตอร์ที่ตรวจยืนยันได้ พร้อมแหล่งที่มา")
    rep.extend("  " + c for c in confirmed) if confirmed else rep.append("  (ไม่มี)")
    rep.append(f"\n[2] Model input shape (ยืนยันจากไฟล์โมเดลจริง)")
    rep.append(f"  {model_shape if model_shape else 'อ่านไม่ได้: ' + str(shape_err)}")
    rep.append(f"  ตรงกับค่าจากโค้ดต้นฉบับ: {'ใช่' if shape_match else 'ไม่ใช่'}")
    rep.append("\n[3] ผลทดสอบแต่ละรายการ")
    for name, st, note in results:
        rep.append(f"  [{st}] {name}" + (f"  -- {note}" if note else ""))
    rep.append("\n[4] จุดที่ยัง BLOCKED (ห้ามเดาค่า - ต้องตรวจสอบก่อน)")
    if critical_blocked:
        rep.extend(f"  - {b}" for b in dict.fromkeys(critical_blocked))
    else:
        rep.append("  (ไม่มี)")
    rep.append("\n[5] การยืนยันความสะอาดของกระบวนการ")
    t8_ok = all(s != "FAILED" for n, s, _ in results if n.startswith("T8"))
    rep.append(f"  - ไม่มีการแก้ไขไฟล์ Day 5-9 (ตรวจจาก mtime ก่อน/หลัง): {'ยืนยัน' if t8_ok else 'ล้มเหลว'}")
    rep.append("  - ไม่มีการเทรนหรือสร้างโมเดลใหม่: ยืนยัน (โหลดแบบ inference เท่านั้น)")
    rep.append("  - ไม่ได้ใช้ adhdata.csv ในการทดสอบ: ยืนยัน (ใช้ข้อมูลสังเคราะห์เฉพาะทดสอบโครงสร้าง)")
    rep.append("  - ผลจากข้อมูลสังเคราะห์ไม่ถูกอ้างเป็นประสิทธิภาพการคัดกรอง: ยืนยัน")
    rep.append("\n" + "=" * 60)
    rep.append(f"DAY 10.2 STATUS: {status}")
    rep.append("=" * 60)
    OUT_SUMMARY.write_text("\n".join(rep) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"DAY 10.2 STATUS: {status}")
    if status == "BLOCKED":
        print("รายการที่ต้องตรวจสอบก่อนไปต่อ:")
        for b in dict.fromkeys(critical_blocked):
            print(f"  - {b}")
    print(f"Report: {OUT_SUMMARY}")
    print("=" * 60)
    sys.exit(0 if status in ("PASSED", "BLOCKED") else 1)


if __name__ == "__main__":
    main()