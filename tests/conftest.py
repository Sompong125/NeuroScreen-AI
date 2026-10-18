"""pytest path setup for NeuroScreen-AI.

The Day 10.2 / 10.3 tests import `day10_02_inference_pipeline` (lives in src/) and
the app imports `eeg_report_utils` (lives next to app.py). The Streamlit app adds
these folders to sys.path itself, but pytest does not, so add them here once for
every test file in this folder. This does not modify or weaken any test.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
