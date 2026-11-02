"""KokoroRVC voice module installer.

Creates a Python venv, installs Kokoro (the speaker) and RVC's inference and training
dependencies, and downloads the pretrained assets RVC needs (HuBERT features, the RMVPE
pitch tracker, and the v2 40k base generator/discriminator). Idempotent: skips work
already done.

Usage:
    python setup.py install
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
RVC_DIR = os.path.join(MODULE_DIR, "rvc")
VENV_DIR = os.path.join(MODULE_DIR, "venv")
DEPS_VERSION = 2
STAMP = os.path.join(VENV_DIR, ".ari_deps_version")

HF = "https://huggingface.co"
ASSETS = {
    # RVC's own pretrained weights.
    "assets/rmvpe/rmvpe.pt":             f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    "assets/pretrained_v2/f0G40k.pth":   f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth",
    "assets/pretrained_v2/f0D40k.pth":   f"{HF}/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth",
    # ContentVec/HuBERT in transformers format (infer/hubert.py loads assets/hubert_base).
    "assets/hubert_base/config.json":                f"{HF}/lengyue233/content-vec-best/resolve/main/config.json",
    "assets/hubert_base/pytorch_model.bin":          f"{HF}/lengyue233/content-vec-best/resolve/main/pytorch_model.bin",
    "assets/hubert_base/preprocessor_config.json":   f"{HF}/facebook/hubert-base-ls960/resolve/main/preprocessor_config.json",
}

PACKAGES = [
    "torch==2.4.1", "torchaudio==2.4.1",
    "numpy>=1.26.4,<2", "scipy>=1.13.1,<2", "librosa>=0.10.2,<0.11", "soundfile>=0.13.0,<1",
    "transformers>=4.49.0,<4.50", "faiss-cpu>=1.13.0,<2", "praat-parselmouth>=0.4.5,<1",
    "torchfcpe>=0.0.4,<0.1", "einops>=0.8.0,<1", "local-attention>=1.11.0,<2", "hyper-connections>=0.1.8,<0.4.3",
    "scikit-learn>=1.6.0,<2", "tensorboard>=2.19.0", "tqdm", "PyYAML", "av>=15.1.0,<16", "ffmpeg-python",
    "matplotlib>=3.8.2,<4", "flask", "kokoro>=0.9.4", "misaki[en]>=0.9.4",
]


def _run(cmd, label):
    print(f"[setup] {label}...", flush=True)
    r = subprocess.run(cmd, cwd=MODULE_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-4000:] + "\n")
        raise RuntimeError(f"{label} failed (exit {r.returncode})")


def _venv_python():
    return os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python")


def _base_python():
    # Kokoro needs 3.10-3.12; RVC's pins are built for 3.11/3.12.
    for name in ("python3.12", "python3.11", "python3.10"):
        found = shutil.which(name) or (f"/opt/homebrew/bin/{name}" if os.path.exists(f"/opt/homebrew/bin/{name}") else None)
        if found:
            return found
    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        return sys.executable
    raise RuntimeError("KokoroRVC needs Python 3.10-3.12; none was found.")


def install():
    if not os.path.exists(_venv_python()):
        _run([_base_python(), "-m", "venv", VENV_DIR], "Creating venv")
    if not (os.path.exists(STAMP) and open(STAMP).read().strip() == str(DEPS_VERSION)):
        _run([_venv_python(), "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools<81", "wheel"], "Upgrading pip")
        _run([_venv_python(), "-m", "pip", "install", "-q", *PACKAGES], "Installing dependencies")
        with open(STAMP, "w") as f:
            f.write(str(DEPS_VERSION))
    else:
        print("[setup] Dependencies already installed.", flush=True)

    for rel, url in ASSETS.items():
        dest = os.path.join(RVC_DIR, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[setup] Downloading {rel}...", flush=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    print("[setup] KokoroRVC ready.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["install"])
    p.add_argument("--module-dir", help="Accepted for ARI's installer; the module always installs next to this file.")
    p.parse_args()
    install()
