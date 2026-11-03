"""KokoroRVC voice module server (ARI voice module contract).

Kokoro speaks the text; RVC then converts it into the trained voice. A voice folder with no
model.pth yet speaks plain Kokoro, so the module runs (and can train) before its first voice exists.

Endpoints: /info /health /synthesise /load_model /train /train/status /train/pause /train/resume /train/cancel
"""

import argparse
import collections
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_DIR = os.path.join(MODULE_DIR, "rvc")
sys.path.insert(0, RVC_DIR)
os.chdir(RVC_DIR)   # RVC resolves assets/ and configs/ relative to its own folder

p = argparse.ArgumentParser()
p.add_argument("--voice-dir", required=True)
p.add_argument("--port", type=int, default=8027)
p.add_argument("--cpu", action="store_true")
p.add_argument("--data-dir")
p.add_argument("--phoneme-subs")
args = p.parse_args()


def _watch_parent(ppid=os.getppid()):
    # Exit with ARI: if the parent goes away, so does this server.
    while True:
        time.sleep(2)
        if os.getppid() != ppid:
            os._exit(0)


threading.Thread(target=_watch_parent, daemon=True).start()

import numpy as np
import soundfile as sf
import librosa
import torch
from flask import Flask, jsonify, request, Response
from kokoro import KPipeline

app = Flask(__name__)
OUT_SR = 24000
kokoro = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
state = {"voice_dir": args.voice_dir, "vc": None, "index": ""}
lock = threading.Lock()


def _settings():
    try:
        with open(os.path.join(state["voice_dir"], "voice_settings.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_rvc(voice_dir, model_file="model.pth"):
    # model_file is model.pth (the latest) or one of the per-save copies, "<voice>_epoch<N>.pth".
    model = os.path.join(voice_dir, model_file)
    state["voice_dir"], state["vc"], state["index"] = voice_dir, None, ""
    if not os.path.exists(model):
        sys.stderr.write(f"[KokoroRVC] no {model_file} in {voice_dir}: speaking plain Kokoro until one is trained\n")
        return
    from configs.config import Config
    from infer.vc.modules import VC
    saved, sys.argv = sys.argv, [sys.argv[0]]
    try:
        cfg = Config()
    finally:
        sys.argv = saved
    os.environ["weight_root"] = voice_dir
    os.environ["rmvpe_root"] = os.path.join(RVC_DIR, "assets", "rmvpe")
    vc = VC(cfg)
    vc.get_vc(model_file)
    state["vc"] = vc
    # The voice's training features, searched in NumPy (RVC's faiss index segfaults inside this process on macOS).
    feats = os.path.join(voice_dir, "model_features.npy")
    state["index"] = feats if os.path.exists(feats) else ""
    sys.stderr.write(f"[KokoroRVC] loaded RVC voice {model_file} from {voice_dir}\n")


_kokoro_f0 = {}


def _kokoro_pitch(voice):
    # Median pitch of a Kokoro voice, measured once on a fixed line.
    if voice not in _kokoro_f0:
        a = np.concatenate([x.numpy() for _, _, x in kokoro("Hey, I'm right here. What do you need? Honestly, you should get some sleep.", voice=voice)])
        f0 = librosa.yin(a.astype(np.float32), fmin=70, fmax=800, sr=OUT_SR)
        f0 = f0[(f0 > 70) & (f0 < 790)]
        _kokoro_f0[voice] = float(np.median(f0)) if len(f0) else 0.0
    return _kokoro_f0[voice]


def _default_pitch(s, kokoro_voice):
    # A pitch in voice_settings.json wins; otherwise shift Kokoro to the trained voice's own pitch.
    if "pitch" in s:
        return float(s["pitch"])
    target, base = s.get("target_f0"), _kokoro_pitch(kokoro_voice)
    return round(12 * np.log2(target / base)) if target and base else 0


def _synth(text, kokoro_voice, speed, pitch, index_rate, protect):
    parts = [a.numpy() if hasattr(a, "numpy") else np.asarray(a) for _, _, a in kokoro(text, voice=kokoro_voice, speed=speed)]
    audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    vc = state["vc"]
    if vc is None:
        return audio, OUT_SR
    from infer.hubert import load_hubert_model
    if vc.hubert_model is None:
        vc.hubert_model = load_hubert_model(vc.config.device, vc.config.is_half)
    audio16 = librosa.resample(audio.astype(np.float32), orig_sr=OUT_SR, target_sr=16000)
    out = vc.pipeline.pipeline(vc.hubert_model, vc.net_g, 0, audio16, [0, 0, 0], int(pitch), "rmvpe",
                               state["index"], index_rate, vc.if_f0, vc.tgt_sr, 0, 1.0, vc.version, protect)
    # RVC voices come out at their training rate (40 kHz here). Everything that plays ARI's voice was built around
    # 24 kHz, so hand back the same.
    out = np.asarray(out, dtype=np.float32) / (32768.0 if np.abs(out).max() > 1.0 else 1.0)
    return librosa.resample(out, orig_sr=vc.tgt_sr, target_sr=OUT_SR), OUT_SR


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/info")
def info():
    s = _settings()
    return jsonify({
        "engine": "KokoroRVC",
        "type": "trainable",
        "parameters": [
            {"name": "speed", "label": "Speed", "min": 0.5, "max": 2.0, "default": s.get("speed", 1.0), "step": 0.05},
            {"name": "pitch", "label": "Pitch (semitones)", "min": -12, "max": 12, "default": _default_pitch(s, s.get("kokoro_voice", "af_heart")), "step": 1},
            {"name": "index_rate", "label": "Voice likeness", "min": 0, "max": 1, "default": s.get("index_rate", 0.75), "step": 0.05},
            {"name": "protect", "label": "Consonant protection", "min": 0, "max": 0.5, "default": s.get("protect", 0.33), "step": 0.01},
        ],
        "training": {"supports_pause": True, "supports_resume": True},
        "features": ["load_model"],
    })


@app.route("/load_model", methods=["POST"])
def load_model():
    d = request.json or {}
    path = d.get("path") or d.get("model_path", "")   # ARI sends "path", as StyleTTS2 reads it
    with lock:
        if os.path.isdir(path):
            _load_rvc(path)
        else:
            _load_rvc(os.path.dirname(path), os.path.basename(path))
    return jsonify({"ok": True, "rvc": state["vc"] is not None})


@app.route("/synthesise", methods=["POST"])
def synthesise():
    d = request.json or {}
    s = _settings()
    text = (d.get("text") or "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400
    kv = d.get("kokoro_voice", s.get("kokoro_voice", "af_heart"))
    with lock:
        wav, sr = _synth(text, kv, float(d.get("speed", s.get("speed", 1.0))),
                         float(d["pitch"]) if "pitch" in d else _default_pitch(s, kv), float(d.get("index_rate", s.get("index_rate", 0.75))),
                         float(d.get("protect", s.get("protect", 0.33))))
    peak = float(np.abs(wav).max()) if len(wav) else 0
    if peak > 1.0:
        wav = wav / peak
    elif wav.dtype != np.float32 and peak > 1:
        wav = wav.astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), mimetype="audio/wav")


# ── Training (same shape as StyleTTS2's, so ARI's training tab works unchanged) ──────────────
train = {"status": "idle", "process": None, "exit_code": None, "log": collections.deque(maxlen=200),
         "progress": {"step": "Idle", "percent": 0, "detail": None}, "args": None}
EPOCH_RX = re.compile(r"Epoch \[(\d+)/(\d+)\]")


def _launch(cmd):
    proc = subprocess.Popen(cmd, cwd=MODULE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid if hasattr(os, "setsid") else None)
    train.update(process=proc, status="running", exit_code=None)

    def watch():
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            train["log"].append(line)
            if m := EPOCH_RX.search(line):
                cur, tot = int(m.group(1)), int(m.group(2))
                # No detail: ARI appends the detail to the log on every poll, so a fixed one repeats.
                train["progress"] = {"step": f"Epoch {cur}/{tot}", "percent": int(100 * cur / tot), "detail": None}
            elif line.startswith("["):
                train["progress"] = {**train["progress"], "step": line.strip("[]")}
        proc.wait()
        train["exit_code"] = proc.returncode
        if train["status"] == "pausing":
            train["status"] = "paused"
        elif train["status"] == "running":
            train["status"] = "completed" if proc.returncode == 0 else "failed"
            if proc.returncode == 0:
                with lock:
                    _load_rvc(state["voice_dir"])

    threading.Thread(target=watch, daemon=True).start()


@app.route("/train", methods=["POST"])
def start_train():
    if train["status"] == "running":
        return jsonify({"error": "Training already in progress"}), 409
    d = request.json or {}
    cmd = [sys.executable, os.path.join(MODULE_DIR, "train.py"), "--audio-dir", d["audio_path"], "--voice-dir", d["voice_dir"],
           "--name", d.get("model_name", "voice"), "--epochs", str(d.get("epochs", 100)), "--save-every", str(d.get("save_every") or 10)]
    if d.get("retrain"):
        cmd.append("--retrain")
    train["args"] = [c for c in cmd if c != "--retrain"]   # resume never wipes
    train["voice_dir"] = d["voice_dir"]   # where train.py looks for the pause request, not this server's own voice
    train["log"].clear()
    _launch(cmd)
    return jsonify({"ok": True})


@app.route("/train/status")
def train_status():
    lines = list(train["log"])
    since = request.args.get("since", type=int)
    return jsonify({"status": train["status"], "progress": train["progress"],
                    "log": lines[since:] if since is not None else lines[-20:], "log_total": len(lines),
                    "exit_code": train["exit_code"]})


def _stop():
    proc = train["process"]
    if proc and proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return True
    return False


def _sentinel():
    return os.path.join(train.get("voice_dir") or state["voice_dir"], ".stop_training")


@app.route("/train/pause", methods=["POST"])
def train_pause():
    # train.py stops at the next checkpoint save, so nothing trained since the last one is lost;
    # resume starts again from that checkpoint.
    if train["status"] != "running":
        return jsonify({"error": "No running training to pause"}), 400
    voice_dir = (train["args"] or [])[train["args"].index("--voice-dir") + 1] if train["args"] else state["voice_dir"]
    with open(os.path.join(voice_dir, ".stop_training"), "w") as f:
        f.write("pause")
    train["status"] = "pausing"
    return jsonify({"ok": True, "message": "Will pause after the next checkpoint is saved"})


@app.route("/train/resume", methods=["POST"])
def train_resume():
    if train["status"] not in ("paused", "pausing") or not train["args"]:
        return jsonify({"error": "Nothing paused to resume"}), 400
    _launch(train["args"])
    return jsonify({"ok": True})


@app.route("/train/cancel", methods=["POST"])
def train_cancel():
    if train["args"]:
        try: os.remove(os.path.join(train["args"][train["args"].index("--voice-dir") + 1], ".stop_training"))
        except OSError: pass
    if _stop():
        train.update(status="failed", exit_code=-15)
        return jsonify({"ok": True})
    return jsonify({"error": "No training process to cancel"}), 400


_load_rvc(args.voice_dir)
app.run(port=args.port)
