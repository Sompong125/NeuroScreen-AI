# ============================================================
# DAY 10.2 - INFERENCE PIPELINE MODULE (REV.2)
# ปรับตามหลักฐานจากโค้ดฝึกจริง (day7_02_prepare_cnn_dataset.py):
#   raw EEG (512,19) -> 50Hz notch (filtfilt) -> per-channel standardize
#   -> scipy.signal.spectrogram (density, psd) -> 10*log10(S+1e-12)
#   -> (19,65,7) float32 -> transpose เป็น (65,7,19)
# - อ่านหลักฐานจากโค้ดต้นฉบับด้วย AST/regex (ไม่รันไฟล์ต้นฉบับ)
# - ไม่เทรน ไม่แก้ไฟล์เดิม ไม่เดาค่า — อะไรยืนยันไม่ได้ = BLOCKED
# ============================================================

import ast
import json
import re
import zipfile
from pathlib import Path

import numpy as np

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent          # .../src/

REFERENCE_FILES = {
    "day5_preprocessing": BASE_DIR / "day5_preprocessing" / "day5_02_preprocessing_test.py",
    "day5_spectrogram":   BASE_DIR / "day5_preprocessing" / "day5_03_spectrogram_test.py",
    "day7_dataset":       BASE_DIR / "day7_cnn" / "day7_02" / "day7_02_prepare_cnn_dataset.py",
    "day8_training":      BASE_DIR / "day8_Model" / "day8_02" / "day8_02_improved_cnn.py",
}
MODEL_PATH = BASE_DIR / "day8_Model" / "day8_02" / "day8_02_improved_model.keras"
NPZ_PATH = BASE_DIR / "day7_cnn" / "day7_02" / "day7_02_cnn_dataset.npz"

THRESHOLD_FIXED = 0.5  # ค่าคงที่ตามกฎโปรเจกต์ — ห้ามเปลี่ยน


class PipelineBlockedError(RuntimeError):
    """เกิดเมื่อมีพารามิเตอร์สำคัญที่ยังยืนยันจากโค้ดต้นฉบับไม่ได้"""
    def __init__(self, blocked_items):
        self.blocked_items = list(dict.fromkeys(blocked_items))
        super().__init__("PIPELINE BLOCKED: " + "; ".join(self.blocked_items))


# ============================================================
# ส่วนที่ 1: อ่านหลักฐานจากโค้ดต้นฉบับ (read-only, ไม่ execute)
# ============================================================

CONST_MAP = {
    "FS": "fs", "SAMPLING_RATE": "fs", "FS_HZ": "fs", "SAMPLE_RATE": "fs",
    "SEGMENT_SIZE": "window_samples",
    "WINDOW_SAMPLES": "window_samples", "SEGMENT_SAMPLES": "window_samples",
    "WINDOW_LEN": "window_samples", "WINDOW_LENGTH": "window_samples",
    "WINDOW_SEC": "window_sec", "WINDOW_SECONDS": "window_sec",
    "SEGMENT_SECONDS": "window_sec", "SEGMENT_SEC": "window_sec",
    "NPERSEG": "nperseg", "NOVERLAP": "noverlap",
    "NFFT": "nfft", "N_FFT": "nfft",
    "N_FREQ": "n_freq", "FREQ_BINS": "n_freq", "N_FREQ_BINS": "n_freq",
    "N_TIME": "n_time", "N_FRAMES": "n_time", "FRAMES": "n_time",
    "N_TIME_BINS": "n_time",
    "N_CHANNELS": "n_channels", "NUM_CHANNELS": "n_channels",
    "CHANNELS": "channels_list", "EEG_CHANNELS": "channels_list",
    "THRESHOLD": "threshold",
    "NOTCH_FREQ": "notch_freq", "NOTCH_FREQUENCY": "notch_freq",
    "NOTCH_Q": "notch_q",
    "FILTER_ORDER": "filter_order", "ORDER": "filter_order",
}

SCAN_PATTERNS = {
    "spectrogram_call": r"spectrogram\(([^)]*)\)",
    "stft_call": r"stft\(([^)]*)\)",
    "db_transform": r"10\.0\s*\*\s*np\.log10\(\s*S\s*\+\s*1e-12\s*\)",
    "iirnotch_used": r"iirnotch\(",
    "filtfilt_used": r"filtfilt\(",
    "standardization": r"\(\s*filtered\s*-\s*mean\s*\)\s*/\s*std",
    "std_zero_guard": r"np\.where\(\s*std\s*==\s*0",
    "nonoverlap_segmentation": r"\[\s*:\s*SEGMENT_SIZE\s*\]",
}


def analyze_reference_file(path):
    """แยก constants / functions จากไฟล์ด้วย AST (ไม่รันโค้ด)"""
    src = path.read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines()
    constants, functions = {}, {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return constants, functions, lines, src, False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, (int, float, str, list, tuple)):
                            constants[t.id] = (val, node.lineno)
                    except Exception:
                        pass
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = ([a.arg for a in node.args.args], node.lineno)
    return constants, functions, lines, src, True


def collect_evidence():
    """รวบรวมหลักฐานพารามิเตอร์จากทุกไฟล์อ้างอิง"""
    evidence = {}
    file_info = {}
    scans = {k: [] for k in SCAN_PATTERNS}

    for name, path in REFERENCE_FILES.items():
        if not path.exists():
            file_info[name] = {"exists": False, "path": str(path)}
            continue
        constants, functions, lines, src, ok = analyze_reference_file(path)
        file_info[name] = {"exists": True, "path": str(path),
                           "functions": functions, "parse_ok": ok}
        for cname, (val, ln) in constants.items():
            key = CONST_MAP.get(cname.upper())
            if key:
                evidence.setdefault(key, []).append(
                    (val, path.name, ln, lines[ln - 1].strip()))
        for key, pat in SCAN_PATTERNS.items():
            for m in re.finditer(pat, src):
                scans[key].append((path.name, m.group(0).replace("\n", " ")))
    return evidence, file_info, scans


def _pick(evidence, key):
    evs = evidence.get(key, [])
    if not evs:
        return None, []
    vals = {v if not isinstance(v, (list, tuple)) else tuple(v) for v, *_ in evs}
    if len(vals) > 1:
        return "CONFLICT", evs
    return evs[0][0], evs


def resolve_parameters(evidence, scans):
    """สรุปค่าพารามิเตอร์จากหลักฐาน: ค่าทุกตัวต้องมีแหล่งอ้างอิง"""
    params, confirmed, blocked = {}, [], []

    def ev_txt(evs):
        return " | ".join(f"{f}:L{l}" for _, f, l, _ in evs)

    v, evs = _pick(evidence, "fs")
    if isinstance(v, (int, float)):
        params["fs"] = int(v); confirmed.append(f"fs = {v}  [{ev_txt(evs)}]")
    else:
        blocked.append("fs (sampling rate) ไม่พบในโค้ดต้นฉบับ")

    v, evs = _pick(evidence, "window_samples")
    if isinstance(v, int):
        params["window_samples"] = v
        confirmed.append(f"window_samples (SEGMENT_SIZE) = {v}  [{ev_txt(evs)}]")
    else:
        blocked.append("SEGMENT_SIZE / window size ยืนยันไม่ได้จากโค้ดต้นฉบับ")

    v, evs = _pick(evidence, "nperseg")
    if isinstance(v, int):
        params["nperseg"] = v; confirmed.append(f"nperseg = {v}  [{ev_txt(evs)}]")
    else:
        blocked.append("nperseg ไม่พบในโค้ดต้นฉบับ")

    v, evs = _pick(evidence, "noverlap")
    if isinstance(v, int):
        params["noverlap"] = v
        confirmed.append(f"noverlap = {v} (hop ภายใน spectrogram)  [{ev_txt(evs)}]")
    else:
        blocked.append("noverlap ไม่พบในโค้ดต้นฉบับ")

    v, evs = _pick(evidence, "channels_list")
    if isinstance(v, (list, tuple)):
        params["n_channels"] = len(v)
        params["channels"] = list(v)
        confirmed.append(f"n_channels = {len(v)}  [{ev_txt(evs)}]")
    else:
        v, evs = _pick(evidence, "n_channels")
        if isinstance(v, int):
            params["n_channels"] = v
            confirmed.append(f"n_channels = {v}  [{ev_txt(evs)}]")
        else:
            blocked.append("จำนวน channel (19) ยืนยันไม่ได้")

    v, evs = _pick(evidence, "notch_freq")
    if isinstance(v, (int, float)):
        params["notch_freq"] = float(v)
        confirmed.append(f"notch_freq = {v} Hz  [{ev_txt(evs)}]")
    else:
        blocked.append("notch frequency ยืนยันไม่ได้")
    v, evs = _pick(evidence, "notch_q")
    if isinstance(v, (int, float)):
        params["notch_q"] = float(v)
        confirmed.append(f"notch_q = {v}  [{ev_txt(evs)}]")
    else:
        blocked.append("notch Q ยืนยันไม่ได้")

    v, evs = _pick(evidence, "threshold")
    if isinstance(v, (int, float)):
        params["threshold"] = float(v)
        confirmed.append(f"threshold = {v}  [{ev_txt(evs)}]")
        if float(v) != THRESHOLD_FIXED:
            blocked.append(f"threshold ในโค้ดคือ {v} ต่างจากค่าบังคับ 0.5")
    else:
        params["threshold"] = THRESHOLD_FIXED
        confirmed.append(f"threshold = {THRESHOLD_FIXED}  [ค่าคงที่ตามกฎโปรเจกต์]")

    v, evs = _pick(evidence, "n_freq")
    if isinstance(v, int):
        params["n_freq"] = v; confirmed.append(f"n_freq = {v}  [{ev_txt(evs)}]")
    elif "nperseg" in params:
        params["n_freq"] = params["nperseg"] // 2 + 1
        confirmed.append(f"n_freq = {params['n_freq']} (derived: nperseg//2 + 1)")
    v, evs = _pick(evidence, "n_time")
    if isinstance(v, int):
        params["n_time"] = v; confirmed.append(f"n_time = {v}  [{ev_txt(evs)}]")
    elif all(k in params for k in ("window_samples", "nperseg", "noverlap")):
        hop = params["nperseg"] - params["noverlap"]
        params["n_time"] = (params["window_samples"] - params["nperseg"]) // hop + 1
        confirmed.append(f"n_time = {params['n_time']} (derived จาก window/nperseg/noverlap)")

    # segmentation mode: หลักฐานจากการหั่น buf[:SEGMENT_SIZE] (ต่อเนื่องไม่ซ้อนทับ)
    if scans.get("nonoverlap_segmentation") and "window_samples" in params:
        params["segmentation"] = "non-overlapping"
        confirmed.append("segmentation = non-overlapping 512-sample chunks  "
                         f"[{scans['nonoverlap_segmentation'][0][0]}: "
                         f"{scans['nonoverlap_segmentation'][0][1]}]")
    else:
        blocked.append("segmentation mode ยืนยันไม่ได้จากโค้ดต้นฉบับ")

    # spectrogram call evidence
    if scans.get("spectrogram_call"):
        fn, call = scans["spectrogram_call"][0]
        confirmed.append(f"spectrogram call = {call[:100]}...  [{fn}]")
        m = re.search(r"scaling\s*=\s*['\"](\w+)['\"]", call)
        params["scaling"] = m.group(1) if m else None
        m = re.search(r"mode\s*=\s*['\"](\w+)['\"]", call)
        params["spec_mode"] = m.group(1) if m else None
        if params.get("scaling") and params.get("spec_mode"):
            confirmed.append(f"spectrogram scaling='{params['scaling']}', mode='{params['spec_mode']}'")
    else:
        blocked.append("ไม่พบการเรียก scipy.signal.spectrogram ในโค้ดต้นฉบับ")

    return params, confirmed, blocked, scans


def readiness(params, scans):
    """เช็กว่าทุกองค์ประกอบของ pipeline ยืนยันได้จากหลักฐานครบหรือยัง"""
    blocked = []
    need = ["fs", "window_samples", "nperseg", "noverlap", "n_channels",
            "n_freq", "n_time", "threshold", "notch_freq", "notch_q"]
    for k in need:
        if k not in params:
            blocked.append(f"parameter '{k}' ยังยืนยันไม่ได้")
    if params.get("segmentation") != "non-overlapping":
        blocked.append("segmentation mode ยังยืนยันไม่ได้")
    if not scans.get("spectrogram_call"):
        blocked.append("ไม่พบการเรียก spectrogram")
    if not scans.get("db_transform"):
        blocked.append("ไม่พบการแปลง 10.0*np.log10(S + 1e-12)")
    if not (scans.get("iirnotch_used") and scans.get("filtfilt_used")):
        blocked.append("ไม่พบการใช้ iirnotch + filtfilt")
    if not (scans.get("standardization") and scans.get("std_zero_guard")):
        blocked.append("ไม่พบ per-channel standardization (พร้อม guard std=0)")
    return blocked


# ============================================================
# ส่วนที่ 2: ตรวจสอบโมเดลจริง (ไม่ใช้ TensorFlow)
# ============================================================

def _find_shape_key(obj):
    if isinstance(obj, dict):
        for k in ("batch_input_shape", "build_input_shape"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        for v in obj.values():
            r = _find_shape_key(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_shape_key(v)
            if r is not None:
                return r
    return None


def peek_model_input_shape(path=MODEL_PATH):
    """อ่าน input shape จากไฟล์ .keras (Keras 3 = ZIP) โดยไม่โหลด TF"""
    try:
        with zipfile.ZipFile(path) as z:
            if "config.json" in z.namelist():
                cfg = json.loads(z.read("config.json"))
                shape = _find_shape_key(cfg)
                if shape is not None:
                    return shape, None
        return None, "ไม่พบ batch_input_shape/build_input_shape ใน config.json"
    except Exception as e:
        return None, str(e)


# ============================================================
# ส่วนที่ 3: ขั้นตอน pipeline (ตรงกับ process_segment ของ day7_02)
# ============================================================

def validate_eeg_input(data, params):
    """ตรวจรูปร่าง/ชนิด/NaN/Inf และจำนวน channel"""
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("EEG input ว่างเปล่า")
    if arr.ndim != 2:
        raise ValueError(f"EEG input ต้องเป็น 2D (samples, channels) ได้ ndim={arr.ndim}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"EEG input ต้องเป็นตัวเลข ได้ dtype={arr.dtype}")
    if np.isnan(arr).any():
        raise ValueError("พบ NaN ใน EEG input")
    if np.isinf(arr).any():
        raise ValueError("พบ Inf ใน EEG input")
    n_ch = params.get("n_channels")
    if n_ch is None:
        raise PipelineBlockedError(["n_channels ยังไม่ยืนยัน"])
    if arr.shape[1] == n_ch:
        x = arr
    elif arr.shape[0] == n_ch:
        x = arr.T
    else:
        raise ValueError(f"จำนวน channel ไม่ตรง: คาดว่า {n_ch}, ได้ shape={arr.shape}")
    win = params.get("window_samples")
    if win is None:
        raise PipelineBlockedError(["window_samples ยังไม่ยืนยัน"])
    if x.shape[0] < win:
        raise ValueError(f"EEG สั้นเกินไป: ต้องการอย่างน้อย {win} samples, ได้ {x.shape[0]}")
    return x.astype(np.float64)


def segment_eeg(x, window_samples):
    """แบ่งแบบไม่ซ้อนทับ (512 ชน 512) — ตรงกับ stream_and_build ของ day7_02
    ส่วนที่ท้ายไม่ถึง 512 ตัวอย่างจะถูกตัดทิ้ง"""
    n_segs = x.shape[0] // window_samples
    if n_segs == 0:
        raise ValueError("ข้อมูลสั้นเกินกว่าจะแบ่งได้แม้แต่ 1 segment (512 samples)")
    return x[:n_segs * window_samples].reshape(n_segs, window_samples, x.shape[1])


def preprocess_segment(seg, params):
    """seg (512,19) -> 50Hz notch (filtfilt) -> per-channel standardization
    (ลอกตรรกะจาก process_segment ใน day7_02_prepare_cnn_dataset.py)"""
    from scipy.signal import filtfilt, iirnotch
    b, a = iirnotch(w0=params["notch_freq"], Q=params["notch_q"], fs=params["fs"])
    filtered = np.empty_like(seg, dtype=np.float64)
    for i in range(seg.shape[1]):
        filtered[:, i] = filtfilt(b, a, seg[:, i])
    mean = filtered.mean(axis=0)
    std = filtered.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (filtered - mean) / std


def make_spectrograms(prepared_segments, params):
    """(N,512,19) -> (N,19,65,7) float32 — ตรงกับ process_segment ของ day7_02"""
    from scipy.signal import spectrogram
    out = []
    for seg in prepared_segments:
        specs = []
        for i in range(seg.shape[1]):
            f, t, S = spectrogram(
                seg[:, i], fs=params["fs"], nperseg=params["nperseg"],
                noverlap=params["noverlap"],
                scaling=params.get("scaling", "density"),
                mode=params.get("spec_mode", "psd"),
            )
            specs.append(10.0 * np.log10(S + 1e-12))
        tensor = np.stack(specs, axis=0)
        expected = (params["n_channels"], params["n_freq"], params["n_time"])
        if tensor.shape != expected:
            raise ValueError(f"Spectrogram shape ไม่ตรง: ได้ {tensor.shape}, คาดว่า {expected}")
        if not np.isfinite(tensor).all():
            raise ValueError("พบ NaN/Inf ใน spectrogram")
        out.append(tensor.astype(np.float32))
    return np.stack(out, axis=0)  # (N, 19, 65, 7)


def to_cnn_input(spec_batch):
    """(N, 19, 65, 7) -> (N, 65, 7, 19) ตามรูปแบบที่ใช้ฝึก"""
    return spec_batch.transpose(0, 2, 3, 1)


_MODEL_CACHE = {}


def load_model_once(path=MODEL_PATH):
    """โหลดโมเดลเพียงครั้งเดียวตลอดการใช้งาน"""
    if "model" not in _MODEL_CACHE:
        from tensorflow import keras
        _MODEL_CACHE["model"] = keras.models.load_model(path, compile=False)
    return _MODEL_CACHE["model"]


def predict_segments(model, X):
    probs = model.predict(X, batch_size=64, verbose=0).flatten()
    if not np.all((probs >= 0.0) & (probs <= 1.0)):
        raise ValueError("ความน่าจะเป็นที่ได้ไม่อยู่ในช่วง [0, 1]")
    return probs


def aggregate_participant(segment_probs, threshold=THRESHOLD_FIXED):
    """รวมผลระดับ participant ด้วยค่าเฉลี่ย + threshold คงที่ 0.5"""
    probs = np.asarray(segment_probs, dtype=float)
    if probs.size == 0:
        raise ValueError("ไม่มี segment probability ให้รวมผล")
    mean_p = float(probs.mean())
    return {
        "mean_probability": mean_p,
        "predicted_class": "ADHD" if mean_p >= threshold else "Control",
        "threshold": threshold,
        "n_segments": int(probs.size),
    }


def run_inference(eeg, params, scans):
    """pipeline เต็ม: validate -> segment -> preprocess -> spectrogram -> CNN -> aggregate"""
    blocked = readiness(params, scans)
    if blocked:
        raise PipelineBlockedError(blocked)
    x = validate_eeg_input(eeg, params)
    segs = segment_eeg(x, params["window_samples"])
    prepared = np.stack([preprocess_segment(s, params) for s in segs], axis=0)
    spec = make_spectrograms(prepared, params)
    X = to_cnn_input(spec)
    model = load_model_once()
    probs = predict_segments(model, X)
    result = aggregate_participant(probs, params["threshold"])
    result["segment_probabilities"] = probs
    result["cnn_input_shape"] = X.shape
    return result