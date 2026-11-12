# LocalCaps

Local, browser-based video caption editor powered by Faster-Whisper.

## Quick start (Windows)

1. Install 64-bit Python 3.11 or 3.12. Make sure the Python launcher (`py`) is available.
2. Extract this ZIP to a folder you can write to.
3. Double-click `run.bat`.
4. The script creates a virtual environment and installs compatible dependencies. The browser opens at `http://127.0.0.1:5000`.
5. Choose a video/audio file, select a Whisper model and generate captions.

`run.bat` automatically uses a compatible NVIDIA GPU when CTranslate2 detects one. If CUDA is unavailable or fails to initialize, LocalCaps falls back to CPU. To force CPU, use `run-cpu.bat`.

## Fix for metadata_errors

This version pins `av==17.1.0`. PyAV 19 removed the `metadata_errors` parameter from `av.open()`, while Faster-Whisper 1.2.1 still passes it. The pin prevents pip from installing the incompatible PyAV release. Do not remove the pin until the Faster-Whisper release in use explicitly supports the newer PyAV API.

## GPU setup

See [GPU_SETUP.md](GPU_SETUP.md) for NVIDIA driver, CUDA 12 and cuDNN 9 requirements and verification commands. NVIDIA drivers and CUDA/cuDNN runtime libraries are not bundled in this ZIP.

## Notes

- The first use of a selected Whisper model downloads its weights. Subsequent runs use the local cache.
- Model downloads and transcription happen locally; model weights are fetched from Hugging Face on first use.
- Large models need substantial RAM/VRAM. Start with `small` to test your setup.
- Keep the application running while using the browser editor.
- This project has not been benchmarked on every GPU, operating system or media codec. Verify exports on your target machine.
