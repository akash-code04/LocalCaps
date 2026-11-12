"""LocalCaps - local Whisper caption server with automatic NVIDIA CUDA detection."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import traceback
import uuid
from pathlib import Path

# Register the bundled CUDA runtime DLL directory BEFORE importing
# CTranslate2/Faster-Whisper. Keep the handles alive for this process.
BASE_DIR = Path(__file__).resolve().parent
CUDA_DLL_DIR = Path(
    os.environ.get(
        "LOCALCAPS_CUDA_LIB_DIR",
        str(BASE_DIR.parent / "cuBLAS.and.cuDNN_CUDA12_win_v3"),
    )
).resolve()
_DLL_DIRECTORY_HANDLES = []

if CUDA_DLL_DIR.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(CUDA_DLL_DIR)))
        # Also expose the directory to dependent native libraries/subprocesses.
        os.environ["PATH"] = str(CUDA_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    except OSError:
        logging.getLogger("localcaps").exception(
            "Could not register CUDA DLL directory: %s", CUDA_DLL_DIR
        )

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import ctranslate2

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("localcaps")

SIZES = ["tiny", "base", "small", "medium", "large-v3"]
PATTERNS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
LANGUAGES = {"auto", "hinglish", "hi", "en", "bn", "mr", "ta", "te", "ur", "es", "fr", "de", "pt", "ar", "ru", "ja", "ko", "zh", "id", "tr"}
MODEL_REPOS = {s: f"Systran/faster-whisper-{s}" for s in SIZES}

models: dict[str, WhisperModel] = {}
JOBS: dict[str, dict] = {}
model_lock = threading.Lock()

_requested_device = os.environ.get("WHISPER_DEVICE", "auto").strip().lower()
_runtime_warning = None
_runtime_device = "cpu"
_runtime_compute = "int8"


def select_runtime():
    """Choose CUDA when requested/available; otherwise use a safe CPU configuration."""
    global _runtime_warning
    if _requested_device == "cpu":
        return "cpu", "int8"
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        count = 0
        _runtime_warning = f"CUDA detection failed; using CPU. Details: {exc}"

    if count > 0:
        return "cuda", "float16"

    if _requested_device == "cuda":
        _runtime_warning = (
            "CUDA was requested but no usable NVIDIA CUDA device was detected. "
            "Using CPU instead. Check the NVIDIA driver, CUDA 12 runtime and cuDNN 9."
        )
    elif _runtime_warning is None:
        _runtime_warning = "No usable NVIDIA CUDA device detected; using CPU."
    return "cpu", "int8"


_runtime_device, _runtime_compute = select_runtime()


def model_path(size: str, download: bool = False) -> str | None:
    """Return cached model path, optionally downloading it if missing."""
    try:
        path = snapshot_download(
            repo_id=MODEL_REPOS[size],
            allow_patterns=PATTERNS,
            local_files_only=not download,
        )
        if Path(path, "model.bin").is_file():
            return path
    except Exception:
        if not download:
            return None
        raise
    return None


def get_model(size: str, job: dict) -> WhisperModel:
    """Load each model once. Fall back to CPU if CUDA libraries fail at load time."""
    global _runtime_device, _runtime_compute, _runtime_warning
    with model_lock:
        if size in models:
            return models[size]

        path = model_path(size, download=False)
        if not path:
            job.update(phase="download", pct=0, msg="Downloading the model for first use...")
            path = model_path(size, download=True)
        if not path:
            raise RuntimeError(f"Model files for '{size}' were not found after download.")

        job.update(phase="load", pct=100, msg="Loading model into memory...")
        try:
            model = WhisperModel(path, device=_runtime_device, compute_type=_runtime_compute)
        except Exception as exc:
            if _runtime_device != "cuda":
                raise
            log.exception("CUDA model initialization failed; retrying on CPU")
            _runtime_warning = f"GPU initialization failed; using CPU instead. Details: {exc}"
            _runtime_device, _runtime_compute = "cpu", "int8"
            model = WhisperModel(path, device="cpu", compute_type="int8")

        models[size] = model
        return model


def transcribe_job(jid: str, audio_path: str, size: str, language: str, max_words: int, prompt: str | None):
    job = JOBS[jid]
    try:
        model = get_model(size, job)
        job.update(phase="transcribe", pct=0, msg="Transcribing audio...")
        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=prompt,
        )

        captions, current = [], []

        def flush():
            if current:
                captions.append({
                    "start": round(float(current[0].start), 3),
                    "end": round(float(current[-1].end), 3),
                    "text": "".join(word.word for word in current).strip(),
                })
                current.clear()

        duration = float(getattr(info, "duration", 0) or 0)
        for segment in segments:
            for word in segment.words or []:
                if current and (
                    len(current) >= max_words
                    or word.start - current[-1].end > 0.7
                ):
                    flush()
                current.append(word)
                spoken = word.word.strip()
                if spoken and spoken[-1] in ".?!।":
                    flush()
            if duration > 0:
                job.update(pct=min(99, int(100 * segment.end / duration)))
        flush()

        job.update(
            phase="done",
            pct=100,
            result={"captions": captions, "language": info.language,
                    "device": _runtime_device, "compute_type": _runtime_compute},
            msg="Transcription complete.",
        )
    except Exception as exc:
        log.error("Transcription job %s failed:\n%s", jid, traceback.format_exc())
        job.update(phase="error", msg=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/runtime")
def runtime_status():
    return jsonify({
        "device": _runtime_device,
        "compute_type": _runtime_compute,
        "cuda_devices": int(ctranslate2.get_cuda_device_count()) if _runtime_device == "cuda" else 0,
        "warning": _runtime_warning,
    })


@app.get("/api/models")
def model_status():
    return jsonify({size: (size in models) or bool(model_path(size)) for size in SIZES})


@app.get("/api/job/<jid>")
def job_status(jid):
    return jsonify(JOBS.get(jid, {"phase": "error", "msg": "Unknown job"}))


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="File is larger than the 2 GB upload limit."), 413


@app.post("/api/transcribe")
def start_transcription():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose a video or audio file first."), 400

    size = request.form.get("model", "small")
    if size not in SIZES:
        return jsonify(error="Invalid Whisper model selection."), 400

    language = request.form.get("language", "auto")
    if language not in LANGUAGES:
        return jsonify(error="Unsupported language selection."), 400
    if language == "auto":
        language = None

    try:
        max_words = int(request.form.get("words", "3"))
    except (TypeError, ValueError):
        return jsonify(error="Words per caption must be a number between 1 and 12."), 400
    if not 1 <= max_words <= 12:
        return jsonify(error="Words per caption must be between 1 and 12."), 400

    prompt = None
    if request.form.get("language") == "hinglish":
        language = "hi"
        prompt = "Hindi and English code-switched speech. Prefer natural Roman-script Hinglish using English letters; preserve English words."

    safe_name = secure_filename(uploaded.filename)
    suffix = Path(safe_name).suffix[:12] or ".media"
    fd, audio_path = tempfile.mkstemp(prefix="localcaps_", suffix=suffix)
    os.close(fd)
    try:
        uploaded.save(audio_path)
    except Exception:
        try:
            os.remove(audio_path)
        except OSError:
            pass
        log.exception("Could not save uploaded media")
        return jsonify(error="Could not save the uploaded file. Check available disk space and permissions."), 500

    jid = uuid.uuid4().hex
    JOBS[jid] = {"phase": "queued", "pct": 0, "msg": "Queued"}
    thread = threading.Thread(
        target=transcribe_job,
        args=(jid, audio_path, size, language, max_words, prompt),
        daemon=True,
        name=f"localcaps-{jid[:8]}",
    )
    thread.start()
    return jsonify(id=jid), 202


@app.post("/api/export")
def export_video():
    """Combine a silent caption-rendered video with the original uploaded audio."""
    rendered = request.files.get("rendered_video")
    original = request.files.get("original_video")

    if rendered is None or not rendered.filename:
        return jsonify(error="The caption-rendered video was not received."), 400
    if original is None or not original.filename:
        return jsonify(error="The original video was not received."), 400

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return jsonify(
            error="FFmpeg was not found. Install FFmpeg and add its bin folder to Windows PATH, then restart LocalCaps."
        ), 503

    original_name = secure_filename(original.filename) or "original_video.mp4"
    original_suffix = Path(original_name).suffix[:12] or ".mp4"

    work_dir = tempfile.mkdtemp(prefix="localcaps_export_")
    rendered_path = os.path.join(work_dir, "captioned_video.webm")
    original_path = os.path.join(work_dir, "original" + original_suffix)
    output_path = os.path.join(work_dir, "captioned_output.mp4")

    try:
        rendered.save(rendered_path)
        original.save(original_path)

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", rendered_path,
            "-i", original_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map_metadata", "1",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60 * 60,
            check=False,
        )

        if result.returncode != 0 or not os.path.isfile(output_path):
            details = (result.stderr or result.stdout or "Unknown FFmpeg error").strip()
            log.error("FFmpeg export failed: %s", details)
            raise RuntimeError(details[-2500:])

        response = send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name=f"{Path(original_name).stem}_captioned.mp4",
        )
        response.call_on_close(lambda: shutil.rmtree(work_dir, ignore_errors=True))
        return response

    except subprocess.TimeoutExpired:
        shutil.rmtree(work_dir, ignore_errors=True)
        return jsonify(error="FFmpeg export timed out after 60 minutes."), 504
    except Exception as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        log.exception("Video export failed")
        return jsonify(error=f"Video export failed: {exc}"), 500


if __name__ == "__main__":
    log.info("LocalCaps runtime: %s (%s)", _runtime_device.upper(), _runtime_compute)
    if CUDA_DLL_DIR.is_dir():
        log.info("CUDA DLL directory registered: %s", CUDA_DLL_DIR)
    else:
        log.warning("CUDA DLL directory not found: %s", CUDA_DLL_DIR)
    if _runtime_warning:
        log.warning(_runtime_warning)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), threaded=True)
