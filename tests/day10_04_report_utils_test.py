"""Unit tests: EEG view, assessment formatting, PDF report (no TensorFlow needed).

Run from project root:  python -m pytest tests/day10_04_report_utils_test.py -v
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import eeg_report_utils as u  # noqa: E402

CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
      "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
FS = 128


def _eeg(n=512 * 3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 19)) * 20 + 300.0


def _result(n_seg=3, probs=None):
    probs = np.array([0.2, 0.3, 0.45]) if probs is None else np.asarray(probs)
    return {
        "predicted_class": "Control", "mean_probability": float(probs.mean()),
        "threshold": 0.5, "n_segments": int(probs.size),
        "segment_probabilities": probs, "cnn_input_shape": (probs.size, 65, 7, 19),
    }


# ---------------- EEG visualization ----------------
def test_view_valid_19_channels_and_figure():
    v = u.build_eeg_view(_eeg(), CH, FS)
    assert v.data.shape == (4 * FS, 19)
    assert v.channel_names == tuple(CH)
    fig = u.plot_eeg_channels(v)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == CH      # order preserved, top->bottom
    assert len(ax.lines) == 19
    assert "µV" not in ax.get_ylabel() and "uV" not in ax.get_ylabel()
    assert "raw units" in ax.get_ylabel()


def test_time_axis_matches_sampling_rate():
    v = u.build_eeg_view(_eeg(), CH, FS)
    assert v.time[0] == 0.0
    assert v.time[1] == pytest.approx(1 / FS)
    assert v.time[-1] == pytest.approx((4 * FS - 1) / FS)
    assert v.seconds_shown == pytest.approx(4.0)
    assert v.is_partial and v.total_seconds == pytest.approx(12.0)


def test_view_uses_raw_values_not_standardized():
    eeg = _eeg()
    v = u.build_eeg_view(eeg, CH, FS)
    np.testing.assert_array_equal(v.data, eeg[:4 * FS])
    assert abs(v.data.mean()) > 100            # offset ~300 retained -> not z-scored


def test_view_caps_seconds():
    v = u.build_eeg_view(_eeg(n=128 * 60), CH, FS, seconds=1000)
    assert v.data.shape[0] == int(u.MAX_SECONDS * FS)


def test_short_recording_not_partial():
    v = u.build_eeg_view(_eeg(n=256), CH, FS)
    assert not v.is_partial and v.data.shape[0] == 256


def test_nan_inf_become_gaps_and_do_not_crash():
    eeg = _eeg()
    eeg[10, 3] = np.nan
    eeg[20, 5] = np.inf
    eeg[30, 7] = -np.inf
    v = u.build_eeg_view(eeg, CH, FS)
    assert v.n_nonfinite == 3
    assert np.isnan(v.data[10, 3]) and np.isnan(v.data[20, 5])
    assert np.isfinite(np.delete(v.data.ravel(), [10 * 19 + 3, 20 * 19 + 5, 30 * 19 + 7])).all()
    u.plot_eeg_channels(v)  # must not raise


@pytest.mark.parametrize("bad", [
    np.zeros((512, 18)), np.zeros(512), np.zeros((2, 512, 19)),
    np.full((512, 19), np.nan), np.zeros((1, 19)), np.array([["a"] * 19] * 600),
    np.zeros((600, 19), dtype=bool),
])
def test_malformed_input_raises_valueerror_not_crash(bad):
    with pytest.raises(ValueError):
        u.build_eeg_view(bad, CH, FS)


def test_no_state_leak_between_calls():
    a = u.build_eeg_view(_eeg(seed=1), CH, FS)
    b = u.build_eeg_view(_eeg(seed=2), CH, FS)
    assert not np.array_equal(a.data, b.data)


# ---------------- Assessment ----------------
def test_assessment_matches_pipeline_values_exactly():
    r = _result()
    s = u.format_assessment(r)
    assert s.predicted_class == r["predicted_class"]
    assert s.mean_probability == r["mean_probability"]
    assert s.threshold == r["threshold"] and s.n_segments == r["n_segments"]
    assert s.percent_text == f"{r['mean_probability'] * 100:.1f}%"


def test_assessment_does_not_recompute_class():
    r = _result()
    r["predicted_class"] = "ADHD"      # deliberately inconsistent: must pass through untouched
    assert u.format_assessment(r).predicted_class == "ADHD"


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("mean_probability"), lambda r: r.pop("predicted_class"),
    lambda r: r.update(mean_probability=float("nan")), lambda r: r.update(mean_probability=1.5),
    lambda r: r.update(n_segments=99), lambda r: r.update(threshold=None),
])
def test_assessment_rejects_incomplete_or_inconsistent(mutate):
    r = _result()
    mutate(r)
    with pytest.raises(ValueError):
        u.format_assessment(r)


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_assessment_rejects_non_dict(bad):
    with pytest.raises(ValueError):
        u.format_assessment(bad)


def test_assessment_tolerates_missing_optional_fields():
    r = _result()
    r.pop("segment_probabilities")
    r.pop("cnn_input_shape")
    s = u.format_assessment(r)
    assert s.segment_probabilities is None and s.cnn_input_shape is None


# ---------------- PDF ----------------
def _pdf_text(data):
    try:
        import pypdf  # optional
    except ImportError:
        return None
    return "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(data)).pages)


def test_pdf_is_valid_two_pages_and_nonempty():
    v = u.build_eeg_view(_eeg(), CH, FS)
    s = u.format_assessment(_result())
    data = u.build_pdf_report(v, s, u.new_report_meta())
    assert data[:5] == b"%PDF-" and b"%%EOF" in data[-1024:]
    assert len(data) > 10_000
    try:
        import pypdf
    except ImportError:
        return
    assert len(pypdf.PdfReader(io.BytesIO(data)).pages) == 2


def test_pdf_contains_results_and_disclaimer_when_text_extractable():
    v = u.build_eeg_view(_eeg(), CH, FS)
    s = u.format_assessment(_result())
    meta = u.new_report_meta()
    text = _pdf_text(u.build_pdf_report(v, s, meta))
    if text is None:
        pytest.skip("pypdf not installed - text content check skipped")
    assert "NeuroScreen AI" in text and meta["report_id"] in text
    assert f"{s.mean_probability:.4f}" in text and s.percent_text in text
    assert "not a medical diagnosis" in text
    assert "first 4.0 s only" in text


def test_pdf_uses_given_summary_without_inference(monkeypatch):
    """build_pdf_report must not import/call the pipeline."""
    sys.modules.pop("day10_02_inference_pipeline", None)
    v = u.build_eeg_view(_eeg(), CH, FS)
    u.build_pdf_report(v, u.format_assessment(_result()), u.new_report_meta())
    assert "day10_02_inference_pipeline" not in sys.modules


def test_pdf_generation_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    v = u.build_eeg_view(_eeg(), CH, FS)
    u.build_pdf_report(v, u.format_assessment(_result()), u.new_report_meta())
    assert list(tmp_path.iterdir()) == []


def test_report_meta_has_no_personal_fields_and_is_unique():
    a, b = u.new_report_meta(), u.new_report_meta()
    assert set(a) == {"report_id", "generated_at"}
    assert a["report_id"] != b["report_id"] and a["report_id"].startswith("NSA-")


def test_pdf_failure_propagates_so_caller_can_handle(monkeypatch):
    v = u.build_eeg_view(_eeg(), CH, FS)
    s = u.format_assessment(_result())
    monkeypatch.setattr(u, "_summary_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        u.build_pdf_report(v, s, u.new_report_meta())
