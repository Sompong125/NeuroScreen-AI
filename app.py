# ============================================================
# DAY 10.3 - STREAMLIT WEB APPLICATION
# NeuroScreen AI - EEG-based ADHD Screening Support
# Research prototype / screening support only
# NOT a medical diagnostic device
# ============================================================

import sys
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR / "src"  # ชี้ไปที่โฟลเดอร์ src

# เพิ่มทั้งโฟลเดอร์หลักและโฟลเดอร์ src ลงใน sys.path
for p in [_APP_DIR, _SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import streamlit as st

from day10_02_inference_pipeline import (
    collect_evidence,
    resolve_parameters,
    readiness,
    run_inference,
    PipelineBlockedError,
)

# ------------------------------------------------------------------
# ค่าคงที่ (ยืนยันแล้วใน Day 10.2 จากโค้ดฝึก Day 5/7/8.2)
# ------------------------------------------------------------------
EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
FS_HZ = 128
MIN_SAMPLES = 512
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # กันแรม: สูงสุด 50 MB

DISCLAIMER = (
    "**⚠️ RESEARCH PROTOTYPE DISCLAIMER**\n\n"
    "NeuroScreen AI เป็น **ต้นแบบสำหรับช่วยคัดกรองในงานวิจัยเท่านั้น**\n\n"
    "- ระบบนี้ **ไม่ใช่** เครื่องมือวินิจฉัยทางการแพทย์\n"
    "- ผลลัพธ์ **ห้าม** ใช้วินิจฉัย ADHD หรือภาวะทางคลินิกใดๆ\n"
    "- ผลลัพธ์ **ห้าม** ใช้แทนการประเมินโดยผู้เชี่ยวชาญ\n"
    "- ผลที่แสดงเป็นผลลัพธ์จากโมเดล ไม่ใช่การยืนยันว่าบุคคลนั้นเป็นหรือไม่เป็น ADHD"
)


def render_disclaimer():
    st.warning(DISCLAIMER)


# ------------------------------------------------------------------
# โหลด pipeline ครั้งเดียวต่อเซสชัน (โมเดลโหลดแบบ lazy ใน run_inference)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline_once():
    try:
        evidence, file_info, scans = collect_evidence()
        params, confirmed, blocked, scans = resolve_parameters(evidence, scans)
        blocked = list(blocked) + list(readiness(params, scans))
        return {
            "ready": len(blocked) == 0,
            "params": params,
            "scans": scans,
            "confirmed": confirmed,
            "blocked": list(dict.fromkeys(blocked)),
        }
    except Exception as e:
        return {
            "ready": False, "params": None, "scans": None,
            "confirmed": [],
            "blocked": [f"Pipeline initialization error: {type(e).__name__}: {e}"],
        }


# ------------------------------------------------------------------
# ตรวจสอบไฟล์ CSV
# ------------------------------------------------------------------
def validate_uploaded_csv(uploaded):
    """คืน (df หรือ None, รายการข้อความผิดพลาด)"""
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        return None, [f"อ่านไฟล์ CSV ไม่ได้: {type(e).__name__}: {e}"]

    if df is None or len(df) == 0:
        return None, ["ไฟล์ CSV ว่างเปล่า (ไม่มีข้อมูล)"]

    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        return None, [
            "ไม่พบคอลัมน์ EEG: " + ", ".join(missing) +
            " — ไฟล์ต้องมีคอลัมน์ EEG ครบทั้ง 19 ช่อง"
        ]

    errors = []
    subset = df[EEG_CHANNELS]
    numeric = subset.apply(pd.to_numeric, errors="coerce")
    n_bad_type = int((numeric.isna() & subset.notna()).sum().sum())
    if n_bad_type > 0:
        errors.append(f"พบค่าที่ไม่ใช่ตัวเลข {n_bad_type} ค่า")
    n_nan = int(numeric.isna().sum().sum()) - n_bad_type
    if n_nan > 0:
        errors.append(f"พบค่าว่าง/NaN {n_nan} ค่า")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        errors.append("พบค่า Inf ในข้อมูล")
    if len(df) < MIN_SAMPLES:
        errors.append(
            f"ข้อมูลสั้นเกินไป: มี {len(df)} samples ต้องการอย่างน้อย "
            f"{MIN_SAMPLES} samples ({MIN_SAMPLES // FS_HZ} วินาที ที่ {FS_HZ} Hz)"
        )
    return (None if errors else df), errors


# ------------------------------------------------------------------
# หน้าเว็บ 5 หน้า
# ------------------------------------------------------------------
def page_home():
    st.title("🧠 NeuroScreen AI")
    st.subheader("EEG-based ADHD Screening Support — Research Prototype")
    render_disclaimer()
    st.markdown("""
### เกี่ยวกับต้นแบบนี้

ระบบนี้ทดลองว่าโมเดล deep learning บนสัญญาณ EEG สามารถ **ช่วยคัดกรอง**
ความเสี่ยงในงานวิจัยได้หรือไม่

**สิ่งที่ระบบทำ**
- รับไฟล์ EEG 19 ช่อง สัญญาณ 128 Hz
- ใช้ preprocessing เดียวกับตอนฝึกโมเดล (50 Hz notch, standardization, spectrogram)
- แสดงความน่าจะเป็นระดับ participant และผลตาม threshold คงที่ 0.5

**สิ่งที่ระบบไม่ทำ**
- ไม่วินิจฉัย ADHD
- ไม่แทนที่การประเมินทางการแพทย์
- ไม่บันทึกหรือส่งต่อไฟล์ที่คุณอัปโหลด (ประมวลผลในหน่วยความจำเท่านั้น)
    """)
    st.info("👉 เริ่มที่เมนู **EEG Input** ทางด้านซ้าย หรืออ่าน **About & Limitations** ก่อน")


def page_eeg_input():
    st.title("📥 EEG Input")
    render_disclaimer()
    st.markdown("""
### รูปแบบข้อมูลที่รองรับ

- **ไฟล์:** CSV (UTF-8)
- **Sampling rate:** 128 Hz
- **คอลัมน์ EEG 19 ช่อง (ชื่อต้องตรง):**
    """)
    st.code(", ".join(EEG_CHANNELS), language=None)
    st.markdown(f"""
- **ความยาวขั้นต่ำ:** {MIN_SAMPLES} samples ({MIN_SAMPLES // FS_HZ} วินาที)
- **ค่าที่รับไม่ได้:** ช่องว่าง, ข้อความ, NaN, Inf
- ไฟล์ถูกประมวลผล **ในหน่วยความจำเท่านั้น ไม่ถูกบันทึกลงดิสก์**
    """)

    uploaded = st.file_uploader("อัปโหลดไฟล์ EEG (.csv)", type=["csv"])
    if uploaded is None:
        st.session_state.pop("current_eeg", None)
        return

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("ไฟล์ใหญ่เกินไป (เกิน 50 MB) กรุณาใช้ไฟล์ที่เล็กลง")
        st.session_state.pop("current_eeg", None)
        return

    with st.spinner("กำลังตรวจสอบโครงสร้างไฟล์..."):
        df, errors = validate_uploaded_csv(uploaded)

    if errors:
        for e in errors:
            st.error(e)
        st.session_state.pop("current_eeg", None)
        return

    eeg = df[EEG_CHANNELS].to_numpy(dtype=np.float64)
    st.session_state["current_eeg"] = eeg
    st.success("✅ โครงสร้างไฟล์ถูกต้อง")
    st.subheader("ตัวอย่างข้อมูล (10 แถวแรก)")
    st.dataframe(df.head(10))
    st.caption(
        f"โหลดแล้ว **{eeg.shape[0]:,} samples** × **{eeg.shape[1]} ช่อง** "
        f"(≈ {eeg.shape[0] / FS_HZ:.1f} วินาที ที่ {FS_HZ} Hz)"
    )
    st.info("ไปที่เมนู **Screening Analysis** เพื่อประมวลผล")


def page_screening():
    st.title("🔬 Screening Analysis")
    render_disclaimer()

    pipe = load_pipeline_once()
    if not pipe["ready"]:
        st.error("Pipeline ถูก **BLOCKED** — ยังยืนยันพารามิเตอร์จากโค้ดฝึกเดิมไม่ครบ")
        st.write("**รายการที่ติด:**")
        for b in pipe["blocked"]:
            st.write(f"- {b}")
        return

    eeg = st.session_state.get("current_eeg")
    if eeg is None:
        st.info("ยังไม่มีข้อมูล — กรุณาอัปโหลดไฟล์ที่เมนู **EEG Input** ก่อน")
        return

    st.markdown(
        f"ข้อมูลพร้อม: **{eeg.shape[0]:,} samples × {eeg.shape[1]} ช่อง** — "
        "ระบบจะใช้ preprocessing เดียวกับตอนฝึกทุกประการ "
        "(notch 50 Hz, standardization, spectrogram nperseg=128/noverlap=64, threshold คงที่ 0.5)"
    )

    if not st.button("▶ รันการวิเคราะห์", type="primary"):
        return

    bar = st.progress(10, text="กำลังแบ่งเซกเมนต์และ preprocessing...")
    try:
        t0 = time.time()
        result = run_inference(eeg, pipe["params"], pipe["scans"])
        elapsed = time.time() - t0
        bar.progress(100, text="เสร็จสิ้น")
        st.session_state["last_result"] = result
        st.success(f"วิเคราะห์เสร็จใน **{elapsed:.2f} วินาที** — ดูผลที่เมนู **Results**")
    except PipelineBlockedError as e:
        bar.empty()
        st.error("Pipeline blocked: " + "; ".join(e.blocked_items))
    except ValueError as e:
        bar.empty()
        st.error(f"ข้อมูลไม่ผ่านการตรวจสอบ: {e}")
    except Exception as e:
        bar.empty()
        st.error(f"ข้อผิดพลาดไม่คาดคิด: {type(e).__name__}: {e}")


def page_results():
    st.title("📊 Results")
    render_disclaimer()

    result = st.session_state.get("last_result")
    if result is None:
        st.info("ยังไม่มีผลลัพธ์ — อัปโหลดไฟล์ที่ **EEG Input** แล้วรันที่ **Screening Analysis** ก่อน")
        return

    mean_p = float(result["mean_probability"])
    pred = result["predicted_class"]
    thr = float(result["threshold"])
    n_seg = int(result["n_segments"])
    seg_probs = np.asarray(result["segment_probabilities"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean probability", f"{mean_p:.4f}")
    c2.metric("ผลตาม threshold", pred)
    c3.metric("Threshold (คงที่)", f"{thr:.2f}")

    st.markdown(
        f"การบันทึกนี้ถูกแบ่งเป็น **{n_seg} เซกเมนต์** (4 วินาที ไม่ซ้อนทับ) "
        f"ความน่าจะเป็นระดับ participant คือค่าเฉลี่ยของเซกเมนต์ทั้งหมด"
    )
    st.subheader("วิธีตีความผลลัพธ์")
    st.markdown(f"""
- ค่าเฉลี่ย `{mean_p:.4f}` คือพฤติกรรมของโมเดลต่อข้อมูลชุดนี้
- ที่ threshold คงที่ `{thr:.2f}` ผลถูกจัดเป็น **{pred}**
- นี่คือ **ผลลัพธ์จากโมเดล ไม่ใช่การวินิจฉัย** — ไม่ยืนยันว่าบุคคลนั้นเป็นหรือไม่เป็น ADHD
- Threshold `{thr:.2f}` เป็นค่าคงที่จากขั้นตอนการวิจัย ไม่ถูกปรับกับข้อมูลนี้
    """)

    st.subheader("ความน่าจะเป็นรายเซกเมนต์")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(seg_probs)), seg_probs, color="steelblue")
    ax.axhline(thr, color="red", linestyle="--", label=f"threshold = {thr:.2f}")
    ax.set_xlabel("Segment index")
    ax.set_ylabel("P(ADHD)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)


def page_about():
    st.title("ℹ️ About & Limitations")
    render_disclaimer()
    st.markdown("""
### ขอบเขตของโปรเจกต์

NeuroScreen AI เป็น **ต้นแบบงานวิจัย** สำหรับช่วยคัดกรอง — ไม่ใช่เครื่องมือแพทย์
และไม่ผ่านการรับรองทางคลินิกใดๆ

### สเปกที่ระบบใช้ (ยืนยันจากโค้ดฝึกเดิมใน Day 10.2)

- **โมเดล:** Improved CNN จาก Day 8.2 (~107K parameters)
- **สัญญาณเข้า:** 19 ช่อง, 128 Hz, แบ่งเซกเมนต์ 512 samples ไม่ซ้อนทับ
- **Preprocessing:** 50 Hz notch (Q=30) ด้วย filtfilt, standardization รายช่อง
- **Spectrogram:** `scipy.signal.spectrogram` nperseg=128, noverlap=64,
  scaling='density', mode='psd' แปลงด้วย `10*log10(S + 1e-12)` (dB)
- **การรวมผล:** ค่าเฉลี่ยความน่าจะเป็นรายเซกเมนต์, threshold คงที่ 0.5

### ข้อจำกัด

1. ชุดทดสอบมีเพียง 18 คน (9 ADHD / 9 Control) — คนเดียวเปลี่ยนตัวเลขได้มาก
2. ใช้ชุดข้อมูลเดียว ไม่มีการทดสอบกับข้อมูลภายนอก
3. ผลทั้งหมดเป็น experimental — ไม่ใช่ประสิทธิภาพทางคลินิก
4. ข้อมูลที่ต่างจากการฝึกมากอาจให้ผลที่เชื่อถือไม่ได้
5. ระบบไม่บันทึก ไม่ส่งต่อ และไม่เปิดเผยข้อมูลส่วนบุคคลใดๆ

### คำแนะนำการใช้งาน

ใช้เพื่อการสำรวจในงานวิจัยและการเรียนการสอนเท่านั้น — ทุกผลลัพธ์ต้องได้รับการ
ตีความโดยผู้เชี่ยวชาญพร้อมตระหนักถึงข้อจำกัดข้างต้นเสมอ
    """)


# ------------------------------------------------------------------
# เมนูนำทาง
# ------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "EEG Input": page_eeg_input,
    "Screening Analysis": page_screening,
    "Results": page_results,
    "About & Limitations": page_about,
}


def main():
    try:
        st.set_page_config(page_title="NeuroScreen AI", page_icon="🧠", layout="wide")
    except Exception:
        pass  # รองรับเมื่อนำไป import ในโหมดทดสอบ

    st.sidebar.title("NeuroScreen AI")
    st.sidebar.caption("Research prototype — screening support only")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))

    pipe = load_pipeline_once()
    if pipe["ready"]:
        st.sidebar.success("Pipeline: ready")
    else:
        st.sidebar.error("Pipeline: blocked")
        with st.sidebar.expander("See blocked items"):
            for b in pipe["blocked"]:
                st.write(f"- {b}")

    PAGES[choice]()


if __name__ == "__main__":
    main()