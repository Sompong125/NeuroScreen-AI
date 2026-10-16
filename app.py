# ============================================================
# DAY 10.3 - STREAMLIT WEB APPLICATION
# NeuroScreen AI - EEG-based ADHD Screening Support
# Research prototype / screening support only
# NOT a medical diagnostic device
#
# UX/UI revision: presentation layer only.
# Unchanged: constants, DISCLAIMER text, load_pipeline_once(),
# validate_uploaded_csv(), run_inference call, session_state keys,
# page names/routing, threshold handling, result fields.
# ============================================================

import base64
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

# ------------------------------------------------------------------
# Design tokens (presentation only)
# ------------------------------------------------------------------
COLOR_BRAND = "#153B66"     # สีน้ำเงินเข้มหลักของเว็บ (ตามโลโก้)
COLOR_PRIMARY = "#0F766E"   # teal (สีเสริม)
COLOR_NAVY = "#12263A"
COLOR_BLUE = COLOR_BRAND

LOGO_PATH = _APP_DIR / "assets" / "logo_mark.png"          # ไอคอนสมอง (favicon)
LOGO_LOCKUP_PATH = _APP_DIR / "assets" / "logo_lockup.png"  # โลโก้เต็ม (ไอคอน + ชื่อ)

APP_CSS = """
<style>
:root {
  --ns-brand: #153B66;
  --ns-brand-dark: #0E2A4A;
  --ns-brand-tint: #E6EEF7;
  --ns-brand-border: #B7CBE3;
  --ns-primary: #0F766E;
  --ns-primary-dark: #0B5A54;
  --ns-blue: #153B66;
  --ns-text: #12263A;
  --ns-muted: #4A5D70;
  --ns-border: #D5E0E8;
  --ns-surface: #FFFFFF;
  --ns-bg-soft: #F2F7FA;
  --ns-radius: 10px;
}

html, body, [class*="css"], .stApp {
  font-family: "Noto Sans Thai", "Leelawadee UI", "Segoe UI", Tahoma,
               "Helvetica Neue", Arial, sans-serif;
  color: var(--ns-text);
}

/* Layout */
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

/* Typography hierarchy */
h1 { font-size: 1.9rem !important; font-weight: 700 !important;
     letter-spacing: -0.01em; margin-bottom: 0.15rem !important; }
h1, h2, h3 { color: var(--ns-brand) !important; }
h2, h3 { font-weight: 650 !important; }
h3 { font-size: 1.15rem !important; margin-top: 1.4rem !important; }
.ns-lead { color: var(--ns-muted); font-size: 1.02rem; margin: 0 0 1rem 0; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--ns-bg-soft);
  border-right: 1px solid var(--ns-border); }
.ns-brand-lockup { display: block; width: 100%; max-width: 230px; height: auto; margin: 0.2rem 0 0.5rem 0; }
.ns-brand { font-size: 1.2rem; font-weight: 700; color: var(--ns-brand); line-height: 1.15; margin-bottom: 0.4rem; }
.ns-brand-sub { font-size: 0.8rem; color: var(--ns-muted); margin-bottom: 1rem; }

/* Status badge (text + symbol, never colour alone) */
.ns-badge { display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
  font-size: 0.82rem; font-weight: 600; border: 1px solid transparent; }
.ns-badge-ok   { background: #E3F4EF; color: #0B5A54; border-color: #9BD3C7; }
.ns-badge-warn { background: #FDECEA; color: #8A1C14; border-color: #F0B3AE; }

/* Disclaimer banner */
.ns-disclaimer { background: #FFF7E6; border: 1px solid #F2C46D;
  border-left: 5px solid #B7791F; border-radius: var(--ns-radius);
  padding: 0.7rem 1rem; margin: 0.6rem 0 0.8rem 0; color: #5B3A06;
  font-size: 0.95rem; }

/* Workflow stepper */
.ns-steps { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.4rem 0 1.2rem 0; }
.ns-step { flex: 1 1 180px; border: 1px solid var(--ns-border);
  border-radius: var(--ns-radius); padding: 0.55rem 0.85rem;
  background: var(--ns-surface); }
.ns-step-title { font-weight: 650; font-size: 0.95rem; }
.ns-step-state { font-size: 0.82rem; color: var(--ns-muted); }
.ns-step-done  { border-color: #9BD3C7; background: #F1FAF7; }
.ns-step-current { border: 2px solid var(--ns-brand); }

/* Cards / containers */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--ns-radius) !important; }
div[data-testid="stMetric"] { padding: 0.2rem 0.2rem; }
div[data-testid="stMetricLabel"] { color: var(--ns-muted); }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 8px;
  font-weight: 600; min-height: 2.6rem; padding: 0.4rem 1.2rem; }

/* Focus visibility (keyboard users) */
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="radio"]:focus-visible, [role="tab"]:focus-visible,
summary:focus-visible, a:focus-visible {
  outline: 3px solid var(--ns-blue) !important; outline-offset: 2px !important; }

/* Respect reduced-motion preference */
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}

/* Sidebar nav buttons (hover / active) */
section[data-testid="stSidebar"] .stButton > button {
  width: 100%; justify-content: flex-start; text-align: left;
  background: transparent; color: var(--ns-text);
  border: 1px solid transparent; box-shadow: none;
  font-weight: 700; margin-bottom: 0.15rem; min-height: 2.8rem;
  transition: background-color 0.15s ease, border-color 0.15s ease;
}
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] p {
  font-weight: 700 !important; font-size: 1.02rem !important; letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--ns-brand-tint); border-color: var(--ns-brand-border); color: var(--ns-brand);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--ns-brand); color: #FFFFFF; border-color: var(--ns-brand-dark);
  font-weight: 700;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: var(--ns-brand-dark); color: #FFFFFF;
}

/* Home: hero */
.ns-hero { position: relative; overflow: hidden; border: 1px solid var(--ns-border);
  border-radius: 16px; padding: 2rem 2rem 1.6rem 2rem; margin-bottom: 0.9rem;
  background: linear-gradient(135deg, #F2F7FA 0%, #E6EEF7 100%); }
.ns-hero-badge { display: inline-block; font-size: 0.8rem; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase; color: var(--ns-brand);
  background: #FFFFFF; border: 1px solid var(--ns-brand-border); border-radius: 999px;
  padding: 0.2rem 0.7rem; margin-bottom: 0.8rem; }
.ns-hero h1 { margin: 0 !important; font-size: 2.2rem !important; line-height: 1; }
.ns-hero-lockup { display: block; height: 84px; width: auto; max-width: 100%; }
.ns-hero-sub { font-size: 1.05rem; font-weight: 600; color: var(--ns-primary-dark); margin: 0.2rem 0 0.6rem 0; }
.ns-hero-text { max-width: 640px; color: var(--ns-muted); margin: 0; }
.ns-hero-wave { position: absolute; right: -10px; bottom: 0; width: 46%; max-width: 420px;
  height: auto; opacity: 0.55; pointer-events: none; }

/* Home: spec chips */
.ns-specs { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.6rem; margin: 0.4rem 0 0.4rem 0; }
.ns-spec { border: 1px solid var(--ns-border); border-radius: var(--ns-radius);
  background: var(--ns-surface); padding: 0.6rem 0.8rem; }
.ns-spec-label { font-size: 0.78rem; color: var(--ns-muted); }
.ns-spec-value { font-size: 1.02rem; font-weight: 700; color: var(--ns-text); }

/* Home: how-it-works cards */
.ns-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.8rem; margin: 0.4rem 0 0.6rem 0; }
.ns-card { border: 1px solid var(--ns-border); border-radius: 12px;
  background: var(--ns-surface); padding: 1rem 1.1rem; }
.ns-card-num { display: inline-flex; align-items: center; justify-content: center;
  width: 1.9rem; height: 1.9rem; border-radius: 50%; background: var(--ns-brand);
  color: #FFFFFF; font-weight: 700; margin-bottom: 0.5rem; }
.ns-card-title { font-weight: 700; margin-bottom: 0.25rem; }
.ns-card-text { color: var(--ns-muted); font-size: 0.93rem; }

@media (max-width: 640px) {
  .ns-hero { padding: 1.3rem 1.1rem; }
  .ns-hero h1 { font-size: 1.7rem !important; }
  .ns-hero-lockup { height: 58px; }
  .ns-hero-wave { display: none; }
}

/* Home: do / don't cards (equal height, green vs red + symbols) */
.ns-duo { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem; align-items: stretch; margin: 0.4rem 0 1rem 0; }
.ns-duo-card { border-radius: 12px; padding: 1.1rem 1.3rem; border: 1px solid; height: 100%; }
.ns-duo-card h4 { margin: 0 0 0.6rem 0; font-size: 1.08rem; font-weight: 700; }
.ns-duo-card ul { margin: 0; padding: 0; list-style: none; }
.ns-duo-card li { position: relative; padding-left: 1.6rem; margin-bottom: 0.5rem; color: var(--ns-text); }
.ns-duo-card li::before { position: absolute; left: 0; top: 0; font-weight: 800; }
.ns-duo-do   { background: #EAF6EF; border-color: #8FCBA6; }
.ns-duo-do h4 { color: #14532D; }
.ns-duo-do li::before { content: "✓"; color: #14532D; }
.ns-duo-dont { background: #FDEDEC; border-color: #F0B3AE; }
.ns-duo-dont h4 { color: #8A1C14; }
.ns-duo-dont li::before { content: "✗"; color: #8A1C14; }

/* About page */
.ns-about-intro { border: 1px solid var(--ns-brand-border); background: var(--ns-brand-tint);
  border-left: 5px solid var(--ns-brand); border-radius: 12px; padding: 1rem 1.3rem;
  margin: 0.4rem 0 0.4rem 0; font-size: 1.02rem; }
.ns-kv { border: 1px solid var(--ns-border); border-radius: 12px; background: var(--ns-surface);
  overflow: hidden; margin: 0.4rem 0 0.4rem 0; }
.ns-kv-row { display: grid; grid-template-columns: 190px 1fr; gap: 1rem;
  padding: 0.8rem 1.2rem; border-bottom: 1px solid var(--ns-border); }
.ns-kv-row:last-child { border-bottom: none; }
.ns-kv-row:nth-child(even) { background: #F8FBFD; }
.ns-kv-key { font-weight: 700; color: var(--ns-brand); }
.ns-kv-val code { background: #EEF3F8; padding: 0.05rem 0.35rem; border-radius: 4px; font-size: 0.9em; }
.ns-limits { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.8rem; margin: 0.4rem 0 0.4rem 0; }
.ns-limit { display: flex; gap: 0.8rem; align-items: flex-start; border: 1px solid #F2C46D;
  background: #FFFBF0; border-radius: 12px; padding: 0.9rem 1.1rem; height: 100%; }
.ns-limit-num { flex: 0 0 1.9rem; height: 1.9rem; border-radius: 50%; background: #B7791F;
  color: #FFFFFF; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; }
.ns-limit-text { color: var(--ns-text); }
.ns-usage { border: 1px solid #9BD3C7; background: #F1FAF7; border-left: 5px solid var(--ns-primary);
  border-radius: 12px; padding: 1rem 1.3rem; margin: 0.4rem 0 1rem 0; }

@media (max-width: 640px) {
  .ns-kv-row { grid-template-columns: 1fr; gap: 0.15rem; }
}

/* Small screens */
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  h1 { font-size: 1.5rem !important; }
}
</style>
"""


def inject_css():
    st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def logo_data_uri(path_str):
    """โหลดโลโก้จากไฟล์ในเครื่องแล้วฝังเป็น data URI (ไม่มีการร้องขอเครือข่ายภายนอก)"""
    try:
        data = Path(path_str).read_bytes()
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def brand_lockup(css_class):
    """โลโก้เต็ม (alt = ชื่อระบบ) ถ้าไม่พบไฟล์ให้ใช้ข้อความแทน"""
    uri = logo_data_uri(str(LOGO_LOCKUP_PATH))
    if uri:
        return f'<img class="{css_class}" src="{uri}" alt="NeuroScreen AI">'
    return "NeuroScreen AI"


def page_header(title, lead=None):
    st.title(title)
    if lead:
        st.markdown(f'<p class="ns-lead">{lead}</p>', unsafe_allow_html=True)


def render_disclaimer(expanded=False):
    """แบนเนอร์สั้นแสดงทุกหน้า + ข้อความ DISCLAIMER ฉบับเต็ม (ไม่ตัดเนื้อหา)"""
    st.markdown(
        '<div class="ns-disclaimer" role="note"><strong>ต้นแบบงานวิจัย — ไม่ใช่เครื่องมือวินิจฉัย</strong> '
        "ผลลัพธ์เป็นข้อมูลสนับสนุนการพิจารณาของผู้เชี่ยวชาญเท่านั้น</div>",
        unsafe_allow_html=True,
    )
    with st.expander("อ่านข้อจำกัดและคำชี้แจงฉบับเต็ม", expanded=expanded):
        st.markdown(DISCLAIMER)


def render_workflow_steps(uploaded=False, valid=False, invalid=False, has_result=False):
    """แสดงขั้นตอนจากสถานะจริงเท่านั้น (ข้อความ + สัญลักษณ์ ไม่ใช้สีอย่างเดียว)"""
    if invalid:
        verify = (False, "ไม่ผ่านการตรวจ ✗")
    else:
        verify = (valid, "ผ่านการตรวจ ✓" if valid else "○ รอไฟล์")
    steps = [
        ("Upload", uploaded, "เลือกไฟล์แล้ว ✓" if uploaded else "○ รอไฟล์"),
        ("Verify", verify[0], verify[1]),
        ("Analyze", has_result, "วิเคราะห์แล้ว ✓" if has_result else "○ ยังไม่ได้วิเคราะห์"),
        ("Results", has_result, "มีผลลัพธ์ ✓" if has_result else "○ ยังไม่มีผลลัพธ์"),
    ]
    html = ['<div class="ns-steps" role="list">']
    for i, (label, done, state) in enumerate(steps, start=1):
        cls = "ns-step ns-step-done" if done else "ns-step"
        html.append(
            f'<div class="{cls}" role="listitem">'
            f'<div class="ns-step-title">{i}. {label}</div>'
            f'<div class="ns-step-state">{state}</div></div>'
        )
    html.append("</div>")
    return "".join(html)


def go_to_page(name):
    """callback สำหรับปุ่มนำทาง (เปลี่ยนเฉพาะหน้าที่แสดง)"""
    st.session_state["page"] = name


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
# ส่วนประกอบของหน้า EEG Input (วิเคราะห์ + ผลลัพธ์ย้ายมารวมในหน้าเดียว)
# ------------------------------------------------------------------
def render_analysis_section(eeg):
    """เนื้อหาเดิมของหน้า Screening Analysis (ตรรกะและการเรียก run_inference เหมือนเดิม)"""
    st.subheader("รันการวิเคราะห์")

    pipe = load_pipeline_once()
    if not pipe["ready"]:
        st.error("Pipeline ถูก **BLOCKED** — ยังยืนยันพารามิเตอร์จากโค้ดฝึกเดิมไม่ครบ")
        st.write("**รายการที่ติด:**")
        for b in pipe["blocked"]:
            st.write(f"- {b}")
        return

    with st.container(border=True):
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
        st.success(f"วิเคราะห์เสร็จใน **{elapsed:.2f} วินาที** — ดูผลด้านล่าง")
    except PipelineBlockedError as e:
        bar.empty()
        st.error("Pipeline blocked: " + "; ".join(e.blocked_items))
    except ValueError as e:
        bar.empty()
        st.error(f"ข้อมูลไม่ผ่านการตรวจสอบ: {e}")
    except Exception as e:
        bar.empty()
        st.error(f"ข้อผิดพลาดไม่คาดคิด: {type(e).__name__}: {e}")


def render_results_section(result):
    """เนื้อหาเดิมของหน้า Results (ใช้ข้อมูลจากผลลัพธ์จริงเท่านั้น)"""
    st.divider()
    st.subheader("ผลลัพธ์")
    with st.expander("คำชี้แจงสำคัญก่อนอ่านผลลัพธ์", expanded=True):
        st.markdown(DISCLAIMER)

    mean_p = float(result["mean_probability"])
    pred = result["predicted_class"]
    thr = float(result["threshold"])
    n_seg = int(result["n_segments"])
    seg_probs = np.asarray(result["segment_probabilities"])

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.metric("Mean probability", f"{mean_p:.4f}")
    with c2:
        with st.container(border=True):
            st.metric("ผลตาม threshold", pred)
    with c3:
        with st.container(border=True):
            st.metric("Threshold (คงที่)", f"{thr:.2f}")

    st.markdown(
        f"การบันทึกนี้ถูกแบ่งเป็น **{n_seg} เซกเมนต์** (4 วินาที ไม่ซ้อนทับ) "
        f"ความน่าจะเป็นระดับ participant คือค่าเฉลี่ยของเซกเมนต์ทั้งหมด"
    )

    st.markdown("#### วิธีตีความผลลัพธ์")
    with st.container(border=True):
        st.markdown(f"""
- ค่าเฉลี่ย `{mean_p:.4f}` คือพฤติกรรมของโมเดลต่อข้อมูลชุดนี้
- ที่ threshold คงที่ `{thr:.2f}` ผลถูกจัดเป็น **{pred}**
- นี่คือ **ผลลัพธ์จากโมเดล ไม่ใช่การวินิจฉัย** — ไม่ยืนยันว่าบุคคลนั้นเป็นหรือไม่เป็น ADHD
- Threshold `{thr:.2f}` เป็นค่าคงที่จากขั้นตอนการวิจัย ไม่ถูกปรับกับข้อมูลนี้
        """)

    st.markdown("#### ความน่าจะเป็นรายเซกเมนต์")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(seg_probs)), seg_probs, color=COLOR_PRIMARY)
    ax.axhline(thr, color=COLOR_BRAND, linestyle="--", linewidth=2.0,
               label=f"threshold = {thr:.2f}",
               path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    ax.set_xlabel("Segment index", fontsize=10)
    ax.set_ylabel("P(ADHD)", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"แท่งแต่ละแท่งคือความน่าจะเป็นของหนึ่งเซกเมนต์ (รวม {n_seg} เซกเมนต์) "
        f"เส้นประคือ threshold คงที่ {thr:.2f}"
    )

    with st.expander("ตารางค่ารายเซกเมนต์ (ข้อมูลเดียวกับกราฟ)"):
        st.dataframe(
            pd.DataFrame({
                "Segment index": np.arange(len(seg_probs)),
                "Probability": seg_probs,
            }),
            width="stretch",
            hide_index=True,
        )


def clear_eeg_state():
    """ล้างข้อมูลและผลลัพธ์ของไฟล์ก่อนหน้า (กันแสดงผลของไฟล์เก่าคู่กับไฟล์ใหม่)"""
    for k in ("current_eeg", "last_result", "eeg_file_sig"):
        st.session_state.pop(k, None)


def _eeg_input_body(flags):
    with st.expander("รูปแบบข้อมูลที่รองรับ", expanded=True):
        st.markdown("""
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
        clear_eeg_state()
        st.info("ยังไม่ได้เลือกไฟล์ — ลากไฟล์ .csv มาวางหรือกดเลือกไฟล์ด้านบน")
        return

    flags["uploaded"] = True

    # ไฟล์เปลี่ยน -> ล้างผลลัพธ์เก่า
    sig = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    if st.session_state.get("eeg_file_sig") != sig:
        st.session_state.pop("last_result", None)
        st.session_state.pop("current_eeg", None)
        st.session_state["eeg_file_sig"] = sig

    if uploaded.size > MAX_UPLOAD_BYTES:
        flags["invalid"] = True
        st.error("ไฟล์ใหญ่เกินไป (เกิน 50 MB) กรุณาใช้ไฟล์ที่เล็กลง")
        st.session_state.pop("current_eeg", None)
        st.session_state.pop("last_result", None)
        return

    with st.spinner("กำลังตรวจสอบโครงสร้างไฟล์..."):
        df, errors = validate_uploaded_csv(uploaded)

    if errors:
        flags["invalid"] = True
        st.error("ไฟล์ไม่ผ่านการตรวจสอบ กรุณาแก้ไขตามรายการด้านล่างแล้วอัปโหลดใหม่")
        for e in errors:
            st.error(e)
        st.session_state.pop("current_eeg", None)
        st.session_state.pop("last_result", None)
        return

    eeg = df[EEG_CHANNELS].to_numpy(dtype=np.float64)
    st.session_state["current_eeg"] = eeg
    flags["valid"] = True
    st.success("✅ โครงสร้างไฟล์ถูกต้อง")

    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        m1.metric("Samples", f"{eeg.shape[0]:,}")
        m2.metric("ช่อง EEG", f"{eeg.shape[1]}")
        m3.metric("ความยาวโดยประมาณ", f"{eeg.shape[0] / FS_HZ:.1f} วินาที")
        st.caption(f"ที่ sampling rate {FS_HZ} Hz")

    st.subheader("ตัวอย่างข้อมูล (10 แถวแรก)")
    st.dataframe(df.head(10), width="stretch")

    st.divider()
    render_analysis_section(eeg)

    result = st.session_state.get("last_result")
    if result is not None:
        flags["has_result"] = True
        render_results_section(result)


# ------------------------------------------------------------------
# หน้าเว็บ 3 หน้า
# ------------------------------------------------------------------
HERO_WAVE_SVG = (
    '<svg class="ns-hero-wave" viewBox="0 0 420 160" aria-hidden="true" focusable="false" '
    'xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round">'
    '<path d="M0 90 C 20 30, 40 150, 60 90 S 100 30, 120 90 S 160 150, 180 90 '
    'S 220 30, 240 90 S 280 150, 300 90 S 340 30, 360 90 S 400 150, 420 90" '
    'stroke="#153B66" stroke-width="3"/>'
    '<path d="M0 120 C 25 80, 45 160, 70 120 S 115 80, 140 120 S 185 160, 210 120 '
    'S 255 80, 280 120 S 325 160, 350 120 S 395 80, 420 120" '
    'stroke="#0F766E" stroke-width="2" opacity="0.7"/>'
    '</svg>'
)


def page_home():
    st.markdown(
        '<section class="ns-hero">'
        '<div class="ns-hero-badge">Research prototype</div>'
        '<h1>' + brand_lockup("ns-hero-lockup") + '</h1>'
        '<p class="ns-hero-sub">EEG-based ADHD Screening Support</p>'
        '<p class="ns-hero-text">ระบบนี้ทดลองว่าโมเดล deep learning บนสัญญาณ EEG สามารถ '
        '<strong>ช่วยคัดกรอง</strong>ความเสี่ยงในงานวิจัยได้หรือไม่ '
        'ผลลัพธ์ใช้เป็นข้อมูลสนับสนุนการพิจารณาของผู้เชี่ยวชาญเท่านั้น</p>'
        + HERO_WAVE_SVG +
        '</section>',
        unsafe_allow_html=True,
    )

    st.button(
        "เริ่มอัปโหลดไฟล์ EEG",
        type="primary",
        key="cta_start",
        on_click=go_to_page,
        args=("EEG Input",),
    )

    render_disclaimer()

    # แผงสรุปสเปก (ค่ามาจากค่าคงที่ในโค้ด) + สถานะ Pipeline จริง
    pipe = load_pipeline_once()
    pipe_label = "พร้อมใช้งาน ✓" if pipe["ready"] else "ถูก BLOCKED ⚠"
    specs = [
        ("ไฟล์ที่รองรับ", "CSV (UTF-8)"),
        ("ช่อง EEG", f"{len(EEG_CHANNELS)} ช่อง"),
        ("Sampling rate", f"{FS_HZ} Hz"),
        ("ความยาวขั้นต่ำ", f"{MIN_SAMPLES // FS_HZ} วินาที ({MIN_SAMPLES} samples)"),
        ("สถานะ Pipeline", pipe_label),
    ]
    st.markdown("### ข้อกำหนดของข้อมูล")
    st.markdown(
        '<div class="ns-specs">'
        + "".join(
            f'<div class="ns-spec"><div class="ns-spec-label">{k}</div>'
            f'<div class="ns-spec-value">{v}</div></div>'
            for k, v in specs
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### ขั้นตอนการใช้งาน")
    st.markdown(
        '<div class="ns-cards">'
        '<div class="ns-card"><div class="ns-card-num">1</div>'
        '<div class="ns-card-title">อัปโหลดไฟล์ EEG</div>'
        '<div class="ns-card-text">เลือกไฟล์ CSV ที่มีคอลัมน์ EEG ครบ 19 ช่อง '
        'ระบบตรวจโครงสร้างไฟล์ให้ทันที</div></div>'
        '<div class="ns-card"><div class="ns-card-num">2</div>'
        '<div class="ns-card-title">รันการวิเคราะห์</div>'
        '<div class="ns-card-text">กดรันในหน้าเดียวกัน ระบบใช้ preprocessing '
        'เดียวกับตอนฝึกโมเดล</div></div>'
        '<div class="ns-card"><div class="ns-card-num">3</div>'
        '<div class="ns-card-title">ให้ผู้เชี่ยวชาญพิจารณา</div>'
        '<div class="ns-card-text">ผลเป็นข้อมูลสนับสนุนจากโมเดลเท่านั้น '
        'ไม่ใช่การวินิจฉัย</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ns-duo">'
        '<div class="ns-duo-card ns-duo-do"><h4>\u2713 สิ่งที่ระบบทำ</h4><ul>'
        '<li>รับไฟล์ EEG 19 ช่อง สัญญาณ 128 Hz</li>'
        '<li>ใช้ preprocessing เดียวกับตอนฝึกโมเดล (50 Hz notch, standardization, spectrogram)</li>'
        '<li>แสดงความน่าจะเป็นระดับ participant และผลตาม threshold คงที่ 0.5</li>'
        '</ul></div>'
        '<div class="ns-duo-card ns-duo-dont"><h4>\u2717 สิ่งที่ระบบไม่ทำ</h4><ul>'
        '<li>ไม่วินิจฉัย ADHD</li>'
        '<li>ไม่แทนที่การประเมินทางการแพทย์</li>'
        '<li>ไม่บันทึกหรือส่งต่อไฟล์ที่คุณอัปโหลด (ประมวลผลในหน่วยความจำเท่านั้น)</li>'
        '</ul></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def page_eeg_input():
    page_header(
        "EEG Input",
        "อัปโหลดไฟล์ ตรวจโครงสร้าง รันการวิเคราะห์ และดูผลลัพธ์ในหน้าเดียว",
    )
    render_disclaimer()

    steps_ph = st.empty()
    flags = {"uploaded": False, "valid": False, "invalid": False, "has_result": False}
    try:
        _eeg_input_body(flags)
    finally:
        steps_ph.markdown(render_workflow_steps(**flags), unsafe_allow_html=True)


def page_about():
    page_header("About & Limitations", "ขอบเขต สเปก และข้อจำกัดของต้นแบบ")
    render_disclaimer()

    st.markdown("### ขอบเขตของโปรเจกต์")
    st.markdown(
        '<div class="ns-about-intro">NeuroScreen AI เป็น <strong>ต้นแบบงานวิจัย</strong> '
        'สำหรับช่วยคัดกรอง — ไม่ใช่เครื่องมือแพทย์ และไม่ผ่านการรับรองทางคลินิกใดๆ</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### สเปกที่ระบบใช้")
    st.caption("ยืนยันจากโค้ดฝึกเดิมใน Day 10.2")
    specs = [
        ("โมเดล", "Improved CNN จาก Day 8.2 (~107K parameters)"),
        ("สัญญาณเข้า", "19 ช่อง, 128 Hz, แบ่งเซกเมนต์ 512 samples ไม่ซ้อนทับ"),
        ("Preprocessing", "50 Hz notch (Q=30) ด้วย filtfilt, standardization รายช่อง"),
        ("Spectrogram",
         "<code>scipy.signal.spectrogram</code> nperseg=128, noverlap=64, "
         "scaling='density', mode='psd' แปลงด้วย <code>10*log10(S + 1e-12)</code> (dB)"),
        ("การรวมผล", "ค่าเฉลี่ยความน่าจะเป็นรายเซกเมนต์, threshold คงที่ 0.5"),
    ]
    st.markdown(
        '<div class="ns-kv">'
        + "".join(
            f'<div class="ns-kv-row"><div class="ns-kv-key">{k}</div>'
            f'<div class="ns-kv-val">{v}</div></div>'
            for k, v in specs
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### ข้อจำกัด")
    limits = [
        "ชุดทดสอบมีเพียง 18 คน (9 ADHD / 9 Control) — คนเดียวเปลี่ยนตัวเลขได้มาก",
        "ใช้ชุดข้อมูลเดียว ไม่มีการทดสอบกับข้อมูลภายนอก",
        "ผลทั้งหมดเป็น experimental — ไม่ใช่ประสิทธิภาพทางคลินิก",
        "ข้อมูลที่ต่างจากการฝึกมากอาจให้ผลที่เชื่อถือไม่ได้",
        "ระบบไม่บันทึก ไม่ส่งต่อ และไม่เปิดเผยข้อมูลส่วนบุคคลใดๆ",
    ]
    st.markdown(
        '<div class="ns-limits">'
        + "".join(
            f'<div class="ns-limit"><div class="ns-limit-num">{i}</div>'
            f'<div class="ns-limit-text">{t}</div></div>'
            for i, t in enumerate(limits, start=1)
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### คำแนะนำการใช้งาน")
    st.markdown(
        '<div class="ns-usage">ใช้เพื่อการสำรวจในงานวิจัยและการเรียนการสอนเท่านั้น — '
        'ทุกผลลัพธ์ต้องได้รับการตีความโดยผู้เชี่ยวชาญพร้อมตระหนักถึงข้อจำกัดข้างต้นเสมอ</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# เมนูนำทาง
# ------------------------------------------------------------------
PAGES = {
    "Home": page_home,
    "EEG Input": page_eeg_input,
    "About & Limitations": page_about,
}


def main():
    try:
        st.set_page_config(
            page_title="NeuroScreen AI",
            page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🧠",
            layout="wide",
        )
    except Exception:
        pass  # รองรับเมื่อนำไป import ในโหมดทดสอบ

    inject_css()

    if st.session_state.get("page") not in PAGES:
        st.session_state["page"] = "Home"
    current = st.session_state["page"]

    st.sidebar.markdown(
        '<div class="ns-brand">' + brand_lockup("ns-brand-lockup") + '</div>'
        '<div class="ns-brand-sub">Research prototype — screening support only</div>',
        unsafe_allow_html=True,
    )
    for name in PAGES:
        is_active = name == current
        st.sidebar.button(
            f"▸ {name}" if is_active else name,
            key=f"nav_{name}",
            type="primary" if is_active else "secondary",
            on_click=go_to_page,
            args=(name,),
            width="stretch",
        )

    pipe = load_pipeline_once()
    if pipe["ready"]:
        st.sidebar.markdown(
            '<span class="ns-badge ns-badge-ok">✓ Pipeline: ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="ns-badge ns-badge-warn">⚠ Pipeline: blocked</span>',
            unsafe_allow_html=True,
        )
        with st.sidebar.expander("See blocked items"):
            for b in pipe["blocked"]:
                st.write(f"- {b}")

    PAGES[current]()


if __name__ == "__main__":
    main()