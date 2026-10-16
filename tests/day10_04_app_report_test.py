"""App-level tests for EEG chart / assessment / PDF using Streamlit AppTest.

The real TensorFlow model is NOT loaded: a stub pipeline module is injected into
sys.modules, returns a fixed result, and counts how many times run_inference ran.
These tests therefore check the UI wiring (no stale results, no repeat inference,
PDF == on-screen result), not model accuracy. Real-model behaviour is covered by
tests/day10_02_pipeline_test.py.

Run from project root:  python -m pytest tests/day10_04_app_report_test.py -v
"""
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from streamlit.testing.v1 import AppTest  # noqa: E402

CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
      "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
APP = str(_ROOT / "app.py")


def _csv(n=512 * 2, seed=0, nan=False, drop_col=None):
    import pandas as pd
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 19)) * 20 + 300.0, columns=CH)
    if nan:
        df.iloc[5, 2] = np.nan
    if drop_col:
        df = df.drop(columns=[drop_col])
    return df.to_csv(index=False).encode("utf-8")


class _Stub:
    calls = 0
    mode = "ok"
    probs = np.array([0.30, 0.40])


@pytest.fixture()
def stub(monkeypatch):
    _Stub.calls, _Stub.mode, _Stub.probs = 0, "ok", np.array([0.30, 0.40])
    m = types.ModuleType("day10_02_inference_pipeline")

    class PipelineBlockedError(RuntimeError):
        def __init__(self, items):
            super().__init__("blocked")
            self.blocked_items = list(items)

    m.PipelineBlockedError = PipelineBlockedError
    m.collect_evidence = lambda: ({}, {}, {})
    m.resolve_parameters = lambda e, s: ({"channels": list(CH)}, [], [], s)
    m.readiness = lambda p, s: []

    def run_inference(eeg, params, scans):
        _Stub.calls += 1
        if _Stub.mode == "blocked":
            raise PipelineBlockedError(["stub blocked"])
        if _Stub.mode == "error":
            raise ValueError("stub bad data")
        pr = _Stub.probs
        mean_p = float(pr.mean())
        return {"predicted_class": "ADHD" if mean_p >= 0.5 else "Control",
                "mean_probability": mean_p, "threshold": 0.5,
                "n_segments": int(pr.size), "segment_probabilities": pr,
                "cnn_input_shape": (pr.size, 65, 7, 19)}

    m.run_inference = run_inference
    monkeypatch.setitem(sys.modules, "day10_02_inference_pipeline", m)
    return _Stub


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["page"] = "EEG Input"
    at.run()
    assert not at.exception, at.exception
    return at


def _upload(at, data, name="a.csv"):
    at.file_uploader[0].set_value((name, data, "text/csv"))
    at.run()
    assert not at.exception, at.exception
    return at


def _run_btn(at):
    return next(b for b in at.button if "รันการวิเคราะห์" in b.label)


def _texts(at):
    out = [e.value for e in at.markdown] + [e.value for e in at.caption]
    out += [e.value for e in at.subheader] + [e.value for e in at.error] + [e.value for e in at.warning]
    return "\n".join(str(x) for x in out)


def _dl_buttons(at):
    return [e for e in at.get("download_button")]


# ---------------- chart ----------------
def test_chart_appears_after_upload_before_analysis(stub):
    at = _upload(_app(), _csv())
    assert "กราฟสัญญาณ EEG" in _texts(at)
    assert at.session_state["eeg_view"].data.shape == (512, 19)
    assert "last_result" not in at.session_state
    assert stub.calls == 0                      # chart must not trigger inference
    assert "วินาทีแรก" in _texts(at) or "ทั้งหมด" in _texts(at)


# ---------------- assessment ----------------
def test_assessment_matches_pipeline_and_shows_percent(stub):
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert not at.exception
    r = at.session_state["last_result"]
    assert stub.calls == 1
    t = _texts(at)
    assert f"{r['mean_probability'] * 100:.1f}%" in t
    assert "Model assessment result" in t
    assert "ไม่ใช่ความมั่นใจทางคลินิก" in t
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Mean probability"] == f"{r['mean_probability']:.4f}"
    assert metrics["ผลตาม threshold"] == r["predicted_class"]


@pytest.mark.parametrize("mode", ["blocked", "error"])
def test_pipeline_block_or_error_shows_message_no_result_no_pdf(stub, mode):
    stub.mode = mode
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert not at.exception
    assert "last_result" not in at.session_state
    assert _dl_buttons(at) == []
    assert len(at.error) >= 1


def test_failed_rerun_does_not_leave_previous_result(stub):
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert "last_result" in at.session_state
    stub.mode = "error"
    _run_btn(at).click().run()
    assert "last_result" not in at.session_state and _dl_buttons(at) == []


def test_new_file_clears_old_result_and_pdf(stub):
    at = _upload(_app(), _csv(seed=1))
    _run_btn(at).click().run()
    assert "report_pdf" in at.session_state
    _upload(at, _csv(seed=2), name="b.csv")
    assert "last_result" not in at.session_state
    assert "report_pdf" not in at.session_state and "report_meta" not in at.session_state
    assert _dl_buttons(at) == []
    assert stub.calls == 1


def test_invalid_file_stops_everything(stub):
    at = _upload(_app(), _csv(nan=True))
    assert len(at.error) >= 1
    assert "eeg_view" not in at.session_state and "last_result" not in at.session_state
    assert not [b for b in at.button if "รันการวิเคราะห์" in b.label]
    at = _upload(_app(), _csv(drop_col="Cz"))
    assert len(at.error) >= 1 and "eeg_view" not in at.session_state


def test_removing_file_clears_state(stub):
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    at.file_uploader[0].set_value(None).run()
    for k in ("eeg_view", "last_result", "report_pdf", "report_meta", "current_eeg"):
        assert k not in at.session_state


# ---------------- PDF ----------------
def test_pdf_button_appears_only_after_valid_result(stub):
    at = _upload(_app(), _csv())
    assert _dl_buttons(at) == []
    _run_btn(at).click().run()
    assert len(_dl_buttons(at)) == 1


def test_pdf_matches_screen_and_no_extra_inference(stub):
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert stub.calls == 1
    at.run(); at.run()                          # simulated reruns / widget interactions
    assert stub.calls == 1                      # no repeated inference
    pdf = at.session_state["report_pdf"]["bytes"]
    assert pdf[:5] == b"%PDF-"
    r = at.session_state["last_result"]
    try:
        import pypdf
    except ImportError:
        pytest.skip("pypdf not installed")
    reader = pypdf.PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 2
    text = "\n".join(p.extract_text() or "" for p in reader.pages)
    assert f"{r['mean_probability']:.4f}" in text
    assert str(r["n_segments"]) in text
    assert at.session_state["report_meta"]["report_id"] in text


def test_pdf_is_rebuilt_for_new_analysis_not_reused(stub):
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    first = at.session_state["report_pdf"]["report_id"]
    stub.probs = np.array([0.9, 0.8])
    _run_btn(at).click().run()
    second = at.session_state["report_pdf"]["report_id"]
    assert first != second
    assert stub.calls == 2


def test_pdf_failure_does_not_crash_app(stub, monkeypatch):
    import eeg_report_utils as u
    monkeypatch.setattr(u, "build_pdf_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert not at.exception
    assert "last_result" in at.session_state        # on-screen result still there
    assert _dl_buttons(at) == []
    assert any("PDF" in e.value for e in at.error)


def test_nothing_written_to_disk(stub, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    at = _upload(_app(), _csv())
    _run_btn(at).click().run()
    assert list(tmp_path.glob("**/*.pdf")) == [] and list(tmp_path.glob("**/*.csv")) == []


# ---------------- regression: existing pages ----------------
@pytest.mark.parametrize("page", ["Home", "EEG Input", "About & Limitations"])
def test_existing_pages_still_render(stub, page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, at.exception
