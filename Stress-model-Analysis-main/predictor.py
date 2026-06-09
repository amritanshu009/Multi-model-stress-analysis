import os
import json
import numpy as np
import joblib
import librosa
import torch
import torch.nn as nn
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BEST_DIR   = os.path.join(os.path.dirname(BASE_DIR), "BEST")

_text_pipe       = None
_audio_pipe      = None
_video_model     = None
_device          = None
_img_tf          = None
_video_num_frames = 12
_video_img_size  = 224

STRESS_WORDS = [
    "stress", "stressed", "overwhelmed", "anxious", "anxiety", "panic",
    "panicking", "can't sleep", "insomnia", "heart racing", "chest tight",
    "terrified", "fear", "hopeless", "breakdown", "can't cope",
    "helpless", "crying", "pressure",
]
CALM_WORDS = [
    "calm", "relaxed", "peaceful", "fine", "okay", "stable",
    "comfortable", "under control", "confident", "happy", "good", "great",
]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_models():
    global _text_pipe, _audio_pipe, _video_model, _device, _img_tf
    global _video_num_frames, _video_img_size

    # Text
    path = os.path.join(BEST_DIR, "stress_text_pipeline.joblib")
    if os.path.exists(path):
        try:
            pipe = joblib.load(path)
            pipe.predict_proba(["test"])        # validate it accepts raw strings
            _text_pipe = pipe
            print("[predictor] Text model loaded (ML pipeline).")
        except Exception as e:
            print(f"[predictor] Text pipeline unusable ({e}), using rule-based fallback.")
    else:
        print("[predictor] stress_text_pipeline.joblib not found, using rule-based fallback.")

    # Audio
    path = os.path.join(BEST_DIR, "stress_audio_pipeline_3class.joblib")
    if os.path.exists(path):
        _audio_pipe = joblib.load(path)
        print("[predictor] Audio model loaded. Classes:", _audio_pipe.named_steps["clf"].classes_)

    # Video config
    cfg_path = os.path.join(BEST_DIR, "stress_video_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        _video_img_size   = cfg.get("IMG_SIZE", 224)
        _video_num_frames = cfg.get("NUM_FRAMES", 12)

    _img_tf = transforms.Compose([
        transforms.Resize((_video_img_size, _video_img_size)),
        transforms.ToTensor(),
    ])

    # Video model
    path = os.path.join(BEST_DIR, "stress_video_resnet18.pt")
    if os.path.exists(path):
        try:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vm = models.resnet18(weights=None)
            vm.fc = nn.Linear(vm.fc.in_features, 2)

            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]

            fixed = {}
            for k, v in state.items():
                k = k.replace("module.", "", 1) if k.startswith("module.") else k
                k = k.replace("backbone.", "", 1) if k.startswith("backbone.") else k
                k = k.replace("classifier.", "fc.", 1) if k.startswith("classifier.") else k
                fixed[k] = v

            vm.load_state_dict(fixed, strict=False)
            vm.to(_device).eval()
            _video_model = vm
            print(f"[predictor] Video model loaded on {_device}.")
        except Exception as e:
            print(f"[predictor] Video model failed to load: {e}")


# ── Individual predictors ──────────────────────────────────────────────────────

def _rule_based_text(text: str) -> float:
    t = text.lower()
    s = sum(1 for w in STRESS_WORDS if w in t)
    c = sum(1 for w in CALM_WORDS   if w in t)
    return float(1 / (1 + 2.71828 ** (-1.2 * (s - c))))


def predict_text(text: str):
    if not text or not text.strip():
        return None
    if _text_pipe is not None:
        try:
            return float(_text_pipe.predict_proba([text])[0, 1])
        except Exception:
            pass
    return _rule_based_text(text)


def _extract_audio_features(path: str, sr: int = 16000) -> np.ndarray:
    y, sr = librosa.load(path, sr=sr)
    y, _  = librosa.effects.trim(y, top_db=25)
    target = int(sr * 1.0)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    zcr    = float(librosa.feature.zero_crossing_rate(y).mean())
    rms    = float(librosa.feature.rms(y=y).mean())
    return np.hstack([mfcc.mean(1), mfcc.std(1), chroma, [zcr, rms]]).astype(np.float32)


def predict_audio(audio_path: str):
    if _audio_pipe is None or not audio_path:
        return None
    try:
        x      = _extract_audio_features(audio_path).reshape(1, -1)
        proba  = _audio_pipe.predict_proba(x)[0]
        classes = list(_audio_pipe.named_steps["clf"].classes_)
        def p(cls): return float(proba[classes.index(cls)]) if cls in classes else 0.0
        return p("High") + 0.5 * p("Moderate")
    except Exception as e:
        print(f"[predictor] Audio prediction error: {e}")
        return None


def _sample_frames(video_path: str, n: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = set(np.linspace(0, total - 1, n).astype(int).tolist())
    frames, cur = {}, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in indices:
            frames[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cur += 1
    cap.release()
    ordered = [frames[i] for i in sorted(indices) if i in frames]
    return ordered or None


@torch.no_grad()
def predict_video(video_path: str):
    if _video_model is None or not video_path:
        return None
    try:
        frames = _sample_frames(video_path, _video_num_frames)
        if frames is None:
            return None
        x      = torch.stack([_img_tf(Image.fromarray(f)) for f in frames]).to(_device)
        probs  = torch.softmax(_video_model(x), dim=1)[:, 1]
        return float(probs.mean().item())
    except Exception as e:
        print(f"[predictor] Video prediction error: {e}")
        return None


# ── Fusion ─────────────────────────────────────────────────────────────────────

def _level(p: float) -> str:
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def predict_fused(text, audio_path, video_path,
                  w_text=0.35, w_audio=0.35, w_video=0.30):
    pt = predict_text(text)          if text        else None
    pa = predict_audio(audio_path)   if audio_path  else None
    pv = predict_video(video_path)   if video_path  else None

    scores, weights, used = [], [], []
    if pt is not None: scores.append(pt); weights.append(w_text);  used.append("text")
    if pa is not None: scores.append(pa); weights.append(w_audio); used.append("audio")
    if pv is not None: scores.append(pv); weights.append(w_video); used.append("video")

    if not scores:
        return {"error": "No valid predictions could be made."}

    w = np.array(weights, dtype=np.float32)
    w /= w.sum()
    p_final = float(np.dot(w, np.array(scores, dtype=np.float32)))

    return {
        "p_text":           round(pt, 4)      if pt is not None else None,
        "p_audio":          round(pa, 4)      if pa is not None else None,
        "p_video":          round(pv, 4)      if pv is not None else None,
        "p_final":          round(p_final, 4),
        "level":            _level(p_final),
        "modalities_used":  used,
    }
