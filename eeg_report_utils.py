# ============================================================
# NeuroScreen AI - EEG visualization, assessment formatting, PDF report
# Presentation layer only. Does NOT preprocess, re-score, or call the model.
# Research prototype / screening support only. NOT a medical diagnostic device.
#
# Dependencies: numpy + matplotlib only (both already in requirements.txt).
# Everything is built in memory; nothing is written to disk.
# ============================================================

from __future__ import annotations

import io
import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

DEFAULT_SECONDS = 4.0
MAX_SECONDS = 10.0  # cap on how much signal is ever drawn

DISCLAIMER_TH_LINES = (
    "NeuroScreen AI เป็นระบบต้นแบบเพื่อสนับสนุนการคัดกรองในงานวิจัยเท่านั้น",
    "ผลลัพธ์จากโมเดลไม่ใช่การวินิจฉัยทางการแพทย์",
    "และไม่สามารถใช้แทนการประเมินโดยผู้เชี่ยวชาญได้",
)
DISCLAIMER_EN_LINES = (
    "NeuroScreen AI is a research prototype intended only to support screening",
    "in research settings. Model output is not a medical diagnosis and cannot",
    "replace assessment by a qualified professional.",
)

_THAI_FONT_CANDIDATES = (
    "Leelawadee UI", "Leelawadee", "Tahoma", "Noto Sans Thai", "Sarabun",
    "TH Sarabun New", "Loma", "Garuda", "Norasi", "Waree",
)


# ------------------------------------------------------------------
# 1) EEG view (raw samples, first N seconds)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class EEGView:
    time: np.ndarray            # seconds, shape (n,)
    data: np.ndarray            # raw units, shape (n, n_channels); non-finite -> NaN
    channel_names: tuple
    fs: float
    n_total_samples: int
    n_nonfinite: int

    @property
    def seconds_shown(self) -> float:
        return float(self.data.shape[0] / self.fs)

    @property
    def total_seconds(self) -> float:
        return float(self.n_total_samples / self.fs)

    @property
    def is_partial(self) -> bool:
        return self.n_total_samples > self.data.shape[0]


def build_eeg_view(eeg, channel_names: Sequence[str], fs: float,
                   seconds: float = DEFAULT_SECONDS,
                   max_seconds: float = MAX_SECONDS) -> EEGView:
    """Take the first `seconds` of the RAW (samples, channels) array.

    Only a small slice is copied (<= max_seconds * fs rows). Non-finite values
    are shown as gaps (NaN) and counted - never replaced by invented numbers.
    Raises ValueError for malformed input.
    """
    arr = np.asarray(eeg)
    if arr.ndim != 2:
        raise ValueError(f"EEG ต้องเป็นอาร์เรย์ 2 มิติ (samples, channels) ได้ ndim={arr.ndim}")
    if not (np.issubdtype(arr.dtype, np.number) and not np.iscomplexobj(arr)):
        raise ValueError(f"EEG ต้องเป็นตัวเลขจริง ได้ dtype={arr.dtype}")
    if arr.shape[1] != len(channel_names):
        raise ValueError(
            f"จำนวนช่องไม่ตรง: ข้อมูล {arr.shape[1]} ช่อง แต่ชื่อช่อง {len(channel_names)} ช่อง")
    if not (fs and fs > 0):
        raise ValueError("sampling rate ต้องมากกว่า 0")

    n_show = min(arr.shape[0], int(round(min(float(seconds), float(max_seconds)) * fs)))
    if n_show < 2:
        raise ValueError("ข้อมูลสั้นเกินกว่าจะวาดกราฟ")

    seg = np.array(arr[:n_show], dtype=np.float64, copy=True)
    bad = ~np.isfinite(seg)
    n_bad = int(bad.sum())
    if n_bad == seg.size:
        raise ValueError("ข้อมูลช่วงที่จะแสดงไม่มีค่าที่เป็นตัวเลขจำกัดเลย")
    seg[bad] = np.nan

    return EEGView(
        time=np.arange(n_show, dtype=np.float64) / float(fs),
        data=seg,
        channel_names=tuple(channel_names),
        fs=float(fs),
        n_total_samples=int(arr.shape[0]),
        n_nonfinite=n_bad,
    )


def plot_eeg_channels(view: EEGView, figsize=(11, 8), title: Optional[str] = None,
                      bottom_margin: float = 0.0) -> Figure:
    """Stacked per-channel traces (same order as channel_names, top to bottom).

    Each trace is centred on its own median FOR DISPLAY ONLY so large DC offsets
    do not hide the signal. Values are raw (not standardized, not filtered).
    Uses matplotlib's Figure directly (no pyplot global state).
    """
    n_ch = view.data.shape[1]
    centered = view.data - np.nanmedian(view.data, axis=0, keepdims=True)
    centered = np.where(np.isnan(centered), np.nan, centered)

    with np.errstate(all="ignore"):
        p_lo = np.nanpercentile(centered, 5, axis=0)
        p_hi = np.nanpercentile(centered, 95, axis=0)
    spread = np.nanmax(p_hi - p_lo) if np.isfinite(p_hi - p_lo).any() else 0.0
    spacing = float(spread) * 1.4 if spread and math.isfinite(spread) else 1.0

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.subplots()
    offsets = (n_ch - 1 - np.arange(n_ch)) * spacing
    for i in range(n_ch):
        ax.plot(view.time, centered[:, i] + offsets[i], linewidth=0.8, color="#153B66")
    ax.set_yticks(offsets)
    ax.set_yticklabels(view.channel_names, fontsize=9)
    ax.set_ylim(offsets[-1] - spacing, offsets[0] + spacing)
    ax.set_xlim(float(view.time[0]), float(view.time[-1]))
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Signal amplitude (raw units), channels offset for display", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(
        title or (f"Raw EEG, first {view.seconds_shown:.1f} s "
                  f"({n_ch} channels, {view.fs:g} Hz)"),
        fontsize=11)
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))
    return fig


# ------------------------------------------------------------------
# 2) Assessment summary (reads the pipeline result; computes nothing new)
# ------------------------------------------------------------------
@dataclass(frozen=True)
class AssessmentSummary:
    predicted_class: str
    mean_probability: float
    percent_text: str          # display of mean_probability only
    threshold: float
    n_segments: int
    segment_probabilities: Optional[np.ndarray]
    cnn_input_shape: Optional[tuple]


def format_assessment(result) -> AssessmentSummary:
    """Validate and unpack a run_inference() result. Raises ValueError if unusable.

    predicted_class / threshold / n_segments are passed through exactly as the
    pipeline returned them. The percentage is mean_probability * 100, nothing else.
    """
    if not isinstance(result, dict):
        raise ValueError("ไม่มีผลการประเมินที่ใช้ได้")
    missing = [k for k in ("predicted_class", "mean_probability", "threshold", "n_segments")
               if result.get(k) is None]
    if missing:
        raise ValueError("ผลการประเมินขาดข้อมูล: " + ", ".join(missing))

    mean_p = float(result["mean_probability"])
    thr = float(result["threshold"])
    if not (math.isfinite(mean_p) and 0.0 <= mean_p <= 1.0):
        raise ValueError("mean_probability ไม่อยู่ในช่วง [0, 1]")
    if not math.isfinite(thr):
        raise ValueError("threshold ไม่ถูกต้อง")
    n_seg = int(result["n_segments"])

    seg = result.get("segment_probabilities")
    seg_arr = None
    if seg is not None:
        seg_arr = np.asarray(seg, dtype=float).ravel()
        if seg_arr.size != n_seg:
            raise ValueError(f"จำนวน segment ไม่สอดคล้อง: n_segments={n_seg}, ค่าจริง {seg_arr.size}")

    shape = result.get("cnn_input_shape")
    return AssessmentSummary(
        predicted_class=str(result["predicted_class"]),
        mean_probability=mean_p,
        percent_text=f"{mean_p * 100:.1f}%",
        threshold=thr,
        n_segments=n_seg,
        segment_probabilities=seg_arr,
        cnn_input_shape=tuple(shape) if shape is not None else None,
    )


# ------------------------------------------------------------------
# 3) PDF report (in memory)
# ------------------------------------------------------------------
def new_report_meta() -> dict:
    """Anonymous reference + timestamp, generated once per analysis run."""
    now = datetime.now().astimezone()
    return {
        "report_id": f"NSA-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}",
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)"),
    }


def _find_thai_font_path() -> Optional[str]:
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None
    for name in _THAI_FONT_CANDIDATES:
        try:
            return fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except Exception:
            continue
    return None


def _summary_page(view: EEGView, summary: AssessmentSummary, meta: dict,
                  thai_font_path: Optional[str]) -> Figure:
    from matplotlib.font_manager import FontProperties

    fig = Figure(figsize=(8.27, 11.69), dpi=100)  # A4 portrait
    fig.patch.set_facecolor("white")

    def put(x, y, s, size=10, weight="normal", color="#12263A", thai=False):
        kw = {}
        if thai and thai_font_path:
            kw["fontproperties"] = FontProperties(fname=thai_font_path, size=size)
        else:
            kw.update(fontsize=size, fontweight=weight)
        fig.text(x, y, s, color=color, va="top", ha="left", **kw)

    y = 0.955
    put(0.08, y, "NeuroScreen AI", size=20, weight="bold", color="#153B66")
    y -= 0.032
    put(0.08, y, "Model Assessment Report (research prototype)", size=11, color="#4A5D70")

    y -= 0.045
    rows = [
        ("Report reference", meta.get("report_id", "-")),
        ("Generated", meta.get("generated_at", "-")),
        ("EEG channels", f"{len(view.channel_names)}"),
        ("Sampling rate", f"{view.fs:g} Hz"),
        ("Recording length", f"{view.n_total_samples} samples ({view.total_seconds:.1f} s)"),
        ("Signal shown in chart",
         f"first {view.seconds_shown:.1f} s only" if view.is_partial
         else f"entire recording ({view.seconds_shown:.1f} s)"),
    ]
    put(0.08, y, "Data summary", size=12, weight="bold", color="#153B66")
    y -= 0.026
    for k, v in rows:
        put(0.08, y, k, size=9.5, color="#4A5D70")
        put(0.34, y, v, size=9.5)
        y -= 0.021

    y -= 0.018
    put(0.08, y, "Model assessment result", size=12, weight="bold", color="#153B66")
    y -= 0.026
    a_rows = [
        ("Model output class", summary.predicted_class),
        ("Mean probability (model output)", f"{summary.mean_probability:.4f}  ({summary.percent_text})"),
        ("Threshold", f"{summary.threshold:.2f}"),
        ("Segments used", f"{summary.n_segments} (4 s each, non-overlapping)"),
    ]
    for k, v in a_rows:
        put(0.08, y, k, size=9.5, color="#4A5D70")
        put(0.42, y, v, size=9.5)
        y -= 0.021
    put(0.08, y,
        "Participant-level probability = mean of segment probabilities. "
        "This is a model output, not clinical confidence.",
        size=8, color="#4A5D70")

    # segment probabilities
    y -= 0.028
    if summary.segment_probabilities is not None and summary.segment_probabilities.size:
        h = 0.20
        ax = fig.add_axes([0.12, y - h - 0.02, 0.80, h])
        sp = summary.segment_probabilities
        ax.bar(np.arange(sp.size), sp, color="#0F766E")
        ax.axhline(summary.threshold, color="#153B66", linestyle="--", linewidth=1.5,
                   label=f"threshold = {summary.threshold:.2f}")
        ax.set_ylim(0, 1.02)
        ax.set_xticks(np.arange(sp.size) if sp.size <= 30 else ax.get_xticks())
        ax.set_xlabel("Segment index", fontsize=8)
        ax.set_ylabel("Model probability", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        y = y - h - 0.075
    else:
        put(0.08, y, "(segment probabilities not available)", size=8.5, color="#4A5D70")
        y -= 0.04

    # disclaimer box
    box_h = 0.19
    fig.patches.append(Rectangle(
        (0.07, y - box_h), 0.86, box_h, transform=fig.transFigure,
        facecolor="#FFF7E6", edgecolor="#B7791F", linewidth=1.0))
    yy = y - 0.014
    put(0.09, yy, "Limitations and medical disclaimer", size=11, weight="bold", color="#5B3A06")
    yy -= 0.028
    if thai_font_path:
        for line in DISCLAIMER_TH_LINES:
            put(0.09, yy, line, size=10, color="#5B3A06", thai=True)
            yy -= 0.022
        yy -= 0.006
    for line in DISCLAIMER_EN_LINES:
        put(0.09, yy, line, size=8.5, color="#5B3A06")
        yy -= 0.018
    if not thai_font_path:
        put(0.09, yy - 0.004, "(Thai font not available on this system; English text shown.)",
            size=7.5, color="#5B3A06")

    put(0.08, 0.025, f"{meta.get('report_id', '')} - contains no personal identifiers",
        size=7.5, color="#4A5D70")
    return fig


def build_pdf_report(view: EEGView, summary: AssessmentSummary, meta: dict) -> bytes:
    """Return PDF bytes (2 pages). Uses the SAME view/summary shown on screen.

    Raises on failure; callers should catch and show a friendly message.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    thai_path = _find_thai_font_path()
    page1 = _summary_page(view, summary, meta, thai_path)
    note = ("first {:.1f} s of {:.1f} s".format(view.seconds_shown, view.total_seconds)
            if view.is_partial else "entire recording")
    page2 = plot_eeg_channels(
        view, figsize=(8.27, 11.69),
        title=f"Raw EEG, {note} ({len(view.channel_names)} channels, {view.fs:g} Hz)",
        bottom_margin=0.03)
    page2.text(0.5, 0.012,
               "Raw values, not filtered/standardized; traces centred per channel for display. "
               "Unit not stated in source data.",
               ha="center", fontsize=7.5, color="#4A5D70")
    with PdfPages(buf) as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)
    data = buf.getvalue()
    if not data.startswith(b"%PDF"):
        raise RuntimeError("PDF generation produced invalid output")
    return data
