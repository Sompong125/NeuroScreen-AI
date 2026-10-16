# ============================================================
# NeuroScreen AI - EEG visualization, assessment formatting, PDF report
# Presentation layer only. Does NOT preprocess, re-score, or call the model.
# Research prototype / screening support only. NOT a medical diagnostic device.
#
# Dependencies: numpy + matplotlib only (both already in requirements.txt).
# Everything is built in memory; nothing is written to disk.
# ============================================================

from __future__ import annotations

import hashlib
import io
import math
import platform
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
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
# 2b) Signal quality indicators (heuristic, scale-free, informational only)
# ------------------------------------------------------------------
FLAT_DIFF_FRACTION = 0.5     # >50% of consecutive-sample differences exactly 0
OUTLIER_ROBUST_Z = 8.0       # |x - median| > 8 * robust sigma counts as "extreme"
OUTLIER_FRACTION_FLAG = 0.01  # flag if >1% of samples are extreme
REL_AMP_HIGH = 4.0           # channel robust sigma > 4x median across channels
REL_AMP_LOW = 0.25           # channel robust sigma < 0.25x median across channels
SEGMENT_BURST_RATIO = 5.0    # segment std > 5x that channel's median segment std

CRITERIA_LINES = (
    "Heuristic checks on the whole uploaded recording; they do not affect the model result.",
    "Flat: >50% of consecutive-sample differences are exactly zero (or zero spread).",
    "Extreme samples: >1% of samples farther than 8 robust sigma (1.4826*MAD) from the median.",
    "Relative amplitude: robust sigma >4x or <0.25x the median across channels.",
    "Segment burst: segment SD >5x that channel's median segment SD (model segmentation).",
    "Scale-free (signal unit unknown). Not validated against expert annotation.",
    "No flags does NOT mean the signal is good; expert review of the raw trace is advised.",
)


@dataclass(frozen=True)
class ChannelQuality:
    name: str
    robust_sigma: float
    flat_fraction: float
    outlier_fraction: Optional[float]
    rel_amplitude: Optional[float]
    flags: tuple


@dataclass(frozen=True)
class SignalQuality:
    channels: tuple
    n_samples: int
    n_segments: int
    window_samples: int
    fs: float
    flagged_segments: tuple     # ((segment_index, (reason, ...)), ...)
    n_nonfinite: int

    @property
    def flagged_channels(self) -> tuple:
        return tuple(c for c in self.channels if c.flags)

    def summary_line(self) -> str:
        return (f"{len(self.flagged_channels)} of {len(self.channels)} channels flagged; "
                f"{len(self.flagged_segments)} of {self.n_segments} segments flagged")


def compute_signal_quality(eeg, channel_names: Sequence[str], fs: float,
                           window_samples: int = 512) -> SignalQuality:
    """Heuristic indicators from the RAW array. Informational only: never gates
    or alters inference. Raises ValueError for malformed input."""
    arr = np.asarray(eeg)
    if arr.ndim != 2:
        raise ValueError(f"EEG ต้องเป็นอาร์เรย์ 2 มิติ ได้ ndim={arr.ndim}")
    if not (np.issubdtype(arr.dtype, np.number) and not np.iscomplexobj(arr)):
        raise ValueError(f"EEG ต้องเป็นตัวเลขจริง ได้ dtype={arr.dtype}")
    if arr.shape[1] != len(channel_names):
        raise ValueError("จำนวนช่องไม่ตรงกับชื่อช่อง")
    if arr.shape[0] < 2 or not (fs and fs > 0) or int(window_samples) < 2:
        raise ValueError("ข้อมูลหรือพารามิเตอร์ไม่ถูกต้อง")

    x = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(x)
    n_bad = int((~finite).sum())
    if not finite.any():
        raise ValueError("ไม่มีค่าที่เป็นตัวเลขจำกัดเลย")
    xm = np.where(finite, x, np.nan)
    n_ch = x.shape[1]

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(xm, axis=0)
        sigma = 1.4826 * np.nanmedian(np.abs(xm - med), axis=0)
        d = np.diff(xm, axis=0)
        n_d = np.maximum(np.isfinite(d).sum(axis=0), 1)
        flat_frac = (d == 0).sum(axis=0) / n_d
        pos = sigma[np.isfinite(sigma) & (sigma > 0)]
        ref = float(np.median(pos)) if pos.size else None

        win = int(window_samples)
        n_seg = x.shape[0] // win
        seg_std = ref_std = None
        if n_seg >= 1:
            seg = xm[:n_seg * win].reshape(n_seg, win, n_ch)
            seg_std = np.nanstd(seg, axis=1)              # (n_seg, n_ch)
            ref_std = np.nanmedian(seg_std, axis=0)       # (n_ch,)

    channels = []
    for i, name in enumerate(channel_names):
        sg = float(sigma[i]) if np.isfinite(sigma[i]) else 0.0
        flags = []
        out_frac = None
        if sg > 0:
            with np.errstate(all="ignore"):
                col = xm[:, i]
                out_frac = float(np.nanmean(np.abs(col - med[i]) > OUTLIER_ROBUST_Z * sg))
        rel = (sg / ref) if (ref and ref > 0) else None
        if flat_frac[i] > FLAT_DIFF_FRACTION or sg == 0:
            flags.append("flat / constant signal")
        if out_frac is not None and out_frac > OUTLIER_FRACTION_FLAG:
            flags.append("frequent extreme samples")
        if rel is not None and sg > 0:
            if rel > REL_AMP_HIGH:
                flags.append("high amplitude vs other channels")
            elif rel < REL_AMP_LOW:
                flags.append("low amplitude vs other channels")
        if (~finite[:, i]).any():
            flags.append("non-finite values present")
        channels.append(ChannelQuality(str(name), sg, float(flat_frac[i]), out_frac, rel, tuple(flags)))

    flagged = []
    if seg_std is not None:
        for k in range(n_seg):
            reasons = []
            bursts, flats = [], []
            for i, name in enumerate(channel_names):
                r = ref_std[i]
                if not (np.isfinite(r) and r > 0):
                    continue
                v = seg_std[k, i]
                if np.isfinite(v) and v == 0:
                    flats.append(str(name))
                elif np.isfinite(v) and v > SEGMENT_BURST_RATIO * r:
                    bursts.append(str(name))
            if bursts:
                reasons.append("amplitude burst: " + ", ".join(bursts))
            if flats:
                reasons.append("flat: " + ", ".join(flats))
            if reasons:
                flagged.append((k, tuple(reasons)))

    return SignalQuality(
        channels=tuple(channels), n_samples=int(x.shape[0]), n_segments=int(n_seg),
        window_samples=win, fs=float(fs), flagged_segments=tuple(flagged), n_nonfinite=n_bad)


# ------------------------------------------------------------------
# 2c) Provenance (what produced this result)
# ------------------------------------------------------------------
NA = "not available"
_PARAM_KEYS = ("fs", "n_channels", "window_samples", "notch_freq", "notch_q", "nperseg",
               "noverlap", "scaling", "spec_mode", "n_freq", "n_time", "threshold")
_LIBS = ("tensorflow", "numpy", "scipy", "matplotlib", "streamlit", "pandas")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def collect_provenance(params, model_path=None, model_sha256: Optional[str] = None,
                       csv_sha256: Optional[str] = None, csv_size_bytes: Optional[int] = None) -> dict:
    """Facts only. Anything unknown is reported as 'not available' - never guessed."""
    from importlib import metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return NA

    p = params if isinstance(params, dict) else {}
    return {
        "model": {
            "file_name": Path(model_path).name if model_path else NA,
            "sha256": model_sha256 or NA,
        },
        "parameters": {k: (p[k] if p.get(k) is not None else NA) for k in _PARAM_KEYS},
        "libraries": {k: ver(k) for k in _LIBS},
        "python": platform.python_version(),
        "input": {
            "csv_sha256": csv_sha256 or NA,
            "csv_size_bytes": csv_size_bytes if csv_size_bytes is not None else NA,
        },
    }


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
                  thai_font_path: Optional[str], quality: Optional[SignalQuality] = None) -> Figure:
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
    if quality is not None:
        rows.append(("Signal quality (heuristic)", quality.summary_line() + " - see page 3"))
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
    for line in DISCLAIMER_EN_LINES:
        put(0.09, yy, line, size=9, color="#5B3A06")
        yy -= 0.019
    if thai_font_path:
        yy -= 0.006
        for line in DISCLAIMER_TH_LINES:
            put(0.09, yy, line, size=9.5, color="#5B3A06", thai=True)
            yy -= 0.021
    if not thai_font_path:
        put(0.09, yy - 0.004, "(Thai font not available on this system; English text shown.)",
            size=7.5, color="#5B3A06")

    put(0.08, 0.025, f"{meta.get('report_id', '')} - contains no personal identifiers",
        size=7.5, color="#4A5D70")
    return fig


def _quality_provenance_page(meta: dict, quality: Optional[SignalQuality],
                             provenance: Optional[dict]) -> Figure:
    fig = Figure(figsize=(8.27, 11.69), dpi=100)
    fig.patch.set_facecolor("white")
    state = {"y": 0.965}

    def put(x, s, size=8.5, weight="normal", color="#12263A", mono=False, dy=0.0165):
        fig.text(x, state["y"], s, fontsize=size, fontweight=weight, color=color,
                 va="top", ha="left", family="monospace" if mono else None)
        state["y"] -= dy

    def head(s):
        state["y"] -= 0.008
        put(0.07, s, size=11.5, weight="bold", color="#153B66", dy=0.026)

    put(0.07, "Signal quality and provenance", size=15, weight="bold", color="#153B66", dy=0.03)
    put(0.07, f"Report {meta.get('report_id', '')}", size=8, color="#4A5D70", dy=0.02)

    if quality is not None:
        head("Signal quality indicators (heuristic)")
        put(0.07, quality.summary_line(), size=9, weight="bold")
        for line in CRITERIA_LINES:
            put(0.07, "- " + line, size=7.3, color="#4A5D70", dy=0.0135)
        state["y"] -= 0.006
        cols = (0.07, 0.17, 0.30, 0.44, 0.58, 0.70)
        for x, t in zip(cols, ("Channel", "Robust SD", "Flat %", "Extreme %", "Rel. amp.", "Flags")):
            fig.text(x, state["y"], t, fontsize=7.8, fontweight="bold", va="top", color="#153B66")
        state["y"] -= 0.017
        for c in quality.channels:
            vals = (c.name, f"{c.robust_sigma:.4g}", f"{c.flat_fraction * 100:.1f}",
                    "n/a" if c.outlier_fraction is None else f"{c.outlier_fraction * 100:.2f}",
                    "n/a" if c.rel_amplitude is None else f"{c.rel_amplitude:.2f}",
                    "; ".join(c.flags) if c.flags else "-")
            for x, t in zip(cols, vals):
                fig.text(x, state["y"], t, fontsize=7.3, va="top",
                         color="#8A1C14" if c.flags and x == cols[-1] else "#12263A")
            state["y"] -= 0.0145
        state["y"] -= 0.008
        if quality.flagged_segments:
            put(0.07, f"Flagged segments ({quality.window_samples / quality.fs:.0f} s each; "
                      f"showing up to 8 of {len(quality.flagged_segments)}):", size=8, weight="bold")
            for k, reasons in quality.flagged_segments[:8]:
                t0 = k * quality.window_samples / quality.fs
                put(0.08, f"#{k} ({t0:.0f}-{t0 + quality.window_samples / quality.fs:.0f} s): "
                          + " | ".join(reasons), size=7.3, dy=0.0135)
        else:
            put(0.07, "No segments flagged by the segment-level checks.", size=8)

    if provenance is not None:
        head("Provenance")
        m, prm = provenance.get("model", {}), provenance.get("parameters", {})
        inp, libs = provenance.get("input", {}), provenance.get("libraries", {})
        put(0.07, f"Model file: {m.get('file_name', NA)}", size=8)
        sha = str(m.get("sha256", NA))
        put(0.07, "Model SHA-256:", size=8, dy=0.0135)
        put(0.09, sha[:32], size=7.3, mono=True, dy=0.0125)
        put(0.09, sha[32:], size=7.3, mono=True, dy=0.016)
        csv_sha = str(inp.get("csv_sha256", NA))
        put(0.07, f"Input CSV: {inp.get('csv_size_bytes', NA)} bytes (file name not recorded)", size=8, dy=0.0135)
        put(0.07, "Input CSV SHA-256:", size=8, dy=0.0135)
        put(0.09, csv_sha[:32], size=7.3, mono=True, dy=0.0125)
        put(0.09, csv_sha[32:], size=7.3, mono=True, dy=0.016)
        put(0.07, "Parameters used by the pipeline:", size=8, weight="bold", dy=0.0145)
        items = [f"{k} = {v}" for k, v in prm.items()]
        for i in range(0, len(items), 2):
            put(0.09, "    ".join(f"{t:<26}" for t in items[i:i + 2]), size=7.3, mono=True, dy=0.0135)
        state["y"] -= 0.004
        put(0.07, "Software: Python " + str(provenance.get("python", NA)) + "; "
            + ", ".join(f"{k} {v}" for k, v in libs.items()), size=7.3, color="#4A5D70", dy=0.0135)

    state["y"] -= 0.012
    put(0.07, "Informational only. Research prototype - not a medical diagnosis.", size=7.5,
        color="#4A5D70")
    return fig


def build_pdf_report(view: EEGView, summary: AssessmentSummary, meta: dict,
                     quality: Optional[SignalQuality] = None,
                     provenance: Optional[dict] = None) -> bytes:
    """Return PDF bytes (2 pages; 3 when quality/provenance are given). Uses the SAME view/summary shown on screen.

    Raises on failure; callers should catch and show a friendly message.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    thai_path = _find_thai_font_path()
    page1 = _summary_page(view, summary, meta, thai_path, quality)
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
    # fonttype 42 embeds TrueType so text (incl. Thai, hashes) stays selectable/searchable
    with matplotlib.rc_context({"pdf.fonttype": 42}), PdfPages(buf) as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)
        if quality is not None or provenance is not None:
            pdf.savefig(_quality_provenance_page(meta, quality, provenance))
    data = buf.getvalue()
    if not data.startswith(b"%PDF"):
        raise RuntimeError("PDF generation produced invalid output")
    return data