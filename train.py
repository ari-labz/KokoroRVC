"""Trains an RVC voice from a folder of clips, for ARI's voice training tab.

Runs RVC's own steps in order (slice and resample, pitch with RMVPE, speech features with
HuBERT, train from the pretrained v2 40k base, build the retrieval index), then copies the
result into the voice folder as model.pth (+ the .index). Progress goes to <voice-dir>/Train.log
in the same shape StyleTTS2 uses: "Epoch [n/total]" lines and LOSS_JSON lines.

Re-running with the same name resumes from the latest checkpoint (RVC keeps one), which is how
pause/resume works: pause stops the process, resume starts it again.

Usage:
    python train.py --audio-dir <wavs> --voice-dir <Voices/KokoroRVC/Name> --name <Name> --epochs 100
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys

RVC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc")
PY = sys.executable
running = {"proc": None}


def _forward_stop(signum, _frame):
    # Cancel from ARI stops this script; take the current RVC step (its own process group) down with it.
    if (proc := running["proc"]) and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _forward_stop)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--voice-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--retrain", action="store_true")
    a = p.parse_args()

    exp = re.sub(r"[^A-Za-z0-9_-]+", "_", a.name)
    exp_dir = os.path.join(RVC_DIR, "logs", exp)
    os.makedirs(a.voice_dir, exist_ok=True)
    log_path = os.path.join(a.voice_dir, "Train.log")
    n_cpu = str(max(1, (os.cpu_count() or 4) // 2))

    def log(line):
        print(line, flush=True)
        with open(log_path, "a", encoding="utf8") as f:
            f.write(line + "\n")

    def step(label, args, on_line=None):
        log(f"[{label}]")
        env = {**os.environ, "PYTHONPATH": RVC_DIR, "PYTHONUNBUFFERED": "1"}   # RVC imports its own packages by top-level name
        # Run as modules (-m train.preprocess), not files: run as a file, the script's own folder comes first on the path
        # and "train" resolves to train/train.py instead of the train package.
        script, rest = args[0], args[1:]
        proc = subprocess.Popen([PY, "-m", script[:-3].replace("/", "."), *rest], cwd=RVC_DIR, env=env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        running["proc"] = proc
        for line in proc.stdout:
            line = line.rstrip()
            if on_line:
                if on_line(line) == "stop":
                    os.killpg(proc.pid, signal.SIGTERM)   # RVC trains in child processes too
                    proc.wait()
                    return
            elif line:
                print(line, flush=True)
        if proc.wait() != 0:
            log(f"{label} failed (exit {proc.returncode}).")
            sys.exit(proc.returncode)

    def save_features():
        # serve.py matches against the voice's training features in NumPy instead of RVC's faiss index.
        import numpy as np
        feats = [np.load(f) for f in sorted(glob.glob(os.path.join(exp_dir, "3_feature768", "*.npy")))]
        if feats:
            np.save(os.path.join(a.voice_dir, "model_features.npy"), np.concatenate(feats, 0).astype(np.float32))
        # The voice's typical pitch (RVC already measured it per frame, in Hz). serve.py shifts Kokoro to it, since
        # RVC keeps the pitch it's given: without this a high voice comes out at Kokoro's lower pitch.
        f0 = [np.load(f) for f in glob.glob(os.path.join(exp_dir, "2b-f0nsf", "*.npy"))]
        voiced = np.concatenate([x[x > 0] for x in f0]) if f0 else np.array([])
        if voiced.size:
            path = os.path.join(a.voice_dir, "voice_settings.json")
            settings = json.load(open(path)) if os.path.exists(path) else {}
            settings["target_f0"] = round(float(np.median(voiced)), 1)
            with open(path, "w") as f:
                json.dump(settings, f, indent=2)

    def export(info, dest="model.pth"):
        # Turns the latest training checkpoint into a playable voice (model.pth, or a named copy), as RVC does at the
        # end of training.
        ckpt = os.path.join(exp_dir, "G_2333333.pth")
        if not os.path.exists(ckpt):
            return
        env = {**os.environ, "PYTHONPATH": RVC_DIR}
        code = ("from train.process_ckpt import extract_small_model as x; "
                f"print(x({ckpt!r}, {exp + '_export'!r}, '40k', 1, {info!r}, 'v2'))")
        subprocess.run([PY, "-c", code], cwd=RVC_DIR, env=env, capture_output=True)
        out = os.path.join(RVC_DIR, "assets", "weights", exp + "_export.pth")
        if os.path.exists(out):
            shutil.copy(out, os.path.join(a.voice_dir, dest))
            if dest == "model.pth":
                save_features()
            log(f"Exported {info} as a playable voice{'' if dest == 'model.pth' else ': ' + dest}.")

    if a.retrain and os.path.isdir(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(RVC_DIR, "assets", "weights"), exist_ok=True)   # RVC saves the final voice here and won't create it

    # The data steps only run once per voice; a resumed run goes straight to training.
    if not glob.glob(os.path.join(exp_dir, "3_feature768", "*.npy")):
        # RVC tries to open every file in the folder as audio, so hand it the audio alone (transcripts often come along).
        audio_only = os.path.join(exp_dir, "_audio")
        shutil.rmtree(audio_only, ignore_errors=True)
        os.makedirs(audio_only)
        for f in os.listdir(a.audio_dir):
            if os.path.splitext(f)[1].lower() in (".wav", ".mp3", ".flac", ".ogg"):
                os.symlink(os.path.join(os.path.abspath(a.audio_dir), f), os.path.join(audio_only, f))
        step("Preparing clips", ["train/preprocess.py", audio_only, "40000", n_cpu, exp_dir, "False", "3.7"])
        step("Extracting pitch", ["train/dataset/extract_f0.py", "cpu", exp_dir, n_cpu, "rmvpe"])
        step("Extracting speech features", ["train/dataset/extract_hubert_feature.py", "cpu", "1", "0", exp_dir, "v2", "False"])

    # One line per clip that made it through every step.
    names = lambda d, ext: {os.path.basename(f)[: -len(ext)] for f in glob.glob(os.path.join(exp_dir, d, "*" + ext))}
    keep = names("0_gt_wavs", ".wav") & names("3_feature768", ".npy") & names("2a_f0", ".wav.npy") & names("2b-f0nsf", ".wav.npy")
    if not keep:
        log("No usable clips after preprocessing.")
        sys.exit(1)
    with open(os.path.join(exp_dir, "filelist.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(
            f"{exp_dir}/0_gt_wavs/{n}.wav|{exp_dir}/3_feature768/{n}.npy|{exp_dir}/2a_f0/{n}.wav.npy|{exp_dir}/2b-f0nsf/{n}.wav.npy|0"
            for n in sorted(keep)))
    if not os.path.exists(os.path.join(exp_dir, "config.json")):
        shutil.copy(os.path.join(RVC_DIR, "configs", "v1", "40k.json"), os.path.join(exp_dir, "config.json"))
    log(f"Training on {len(keep)} clips for {a.epochs} epochs.")

    epoch_rx = re.compile(r"====>\s*\S+?[:：]\s*(\d+)")
    loss_rx = re.compile(r"loss_gen=([\d.]+), loss_fm=([\d.]+),\s*loss_mel=([\d.]+), loss_kl=([\d.]+)")
    saved_rx = re.compile(r"Saving model and optimizer state at epoch (\d+) to .*D_")
    sums = []   # (gen, fm, mel, kl) for each logged step this epoch
    sentinel = os.path.join(a.voice_dir, ".stop_training")
    if os.path.exists(sentinel):
        os.remove(sentinel)
    paused = {"at": None, "saved": False}

    def on_train_line(line):
        if m := loss_rx.search(line):
            sums.append(tuple(float(x) for x in m.groups()))
        if saved_rx.search(line):
            paused["saved"] = True
        if m := epoch_rx.search(line):
            ep = int(m.group(1))
            log(f"Epoch [{ep}/{a.epochs}]")
            # RVC keeps only its latest checkpoint, so keep a playable copy of every save, named by epoch, to compare later.
            if paused["saved"]:
                export(f"epoch {ep}", f"{a.name}_epoch{ep}.pth")
            if sums:
                gen, fm, mel, kl = (round(sum(c) / len(c), 4) for c in zip(*sums))
                # The control panel's chart has four slots named after StyleTTS2's losses; it relabels them per engine.
                log("LOSS_JSON: " + json.dumps({"epoch": ep, "val": mel, "f0": gen, "sty": fm, "diff": kl, "engine": "KokoroRVC"}))
                sums.clear()
            # Pause: RVC logs "Saving ..." before it writes and the epoch line after, so stopping on the epoch line
            # that follows a save never cuts a checkpoint short, and resume loses nothing.
            if paused["saved"] and ep < a.epochs and os.path.exists(sentinel):   # on the last epoch, just finish
                paused["at"] = ep
                return "stop"
            paused["saved"] = False
        elif line:
            print(line, flush=True)

    step("Training", ["train/train.py", "-e", exp, "-sr", "40k", "-f0", "1", "-bs", str(a.batch_size),
                      "-te", str(a.epochs), "-se", str(a.save_every),
                      "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
                      "-l", "1", "-c", "0", "-sw", "0", "-v", "v2"], on_train_line)

    if paused["at"] is not None:
        os.remove(sentinel)
        export(f"epoch {paused['at']}")   # so a paused voice can be set active and listened to
        log(f"── Paused at epoch {paused['at']} · checkpoint saved ──")
        return

    step("Building index", ["train/train_index.py", exp, "v2", a.voice_dir, n_cpu])

    weight = os.path.join(RVC_DIR, "assets", "weights", exp + ".pth")
    if not os.path.exists(weight):
        log(f"Training finished but {weight} is missing.")
        sys.exit(1)
    shutil.copy(weight, os.path.join(a.voice_dir, "model.pth"))
    save_features()
    added = sorted(glob.glob(os.path.join(exp_dir, "added_*.index")) + glob.glob(os.path.join(a.voice_dir, "*added_*.index")), key=os.path.getmtime)
    if added:
        shutil.copy(added[-1], os.path.join(a.voice_dir, "model.index"))
    log(f"Done: {a.voice_dir}/model.pth")


if __name__ == "__main__":
    main()
