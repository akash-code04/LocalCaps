# NVIDIA GPU setup (Windows)

LocalCaps automatically selects an NVIDIA GPU when CTranslate2 can see a compatible CUDA device. It uses `float16` on CUDA and `int8` on CPU. If GPU initialization fails, it logs the reason and falls back to CPU rather than stopping the application.

## Requirements

1. A supported NVIDIA GPU.
2. A current NVIDIA display driver.
3. CUDA 12 runtime libraries and cuDNN 9 compatible with CTranslate2. Install the CUDA/cuDNN versions required by the installed CTranslate2 release and make their `bin` directories available on `PATH`.
4. Python 3.11 or 3.12 (64-bit) is recommended.

The Python installer cannot install or bundle NVIDIA's driver, CUDA toolkit, or cuDNN. Install those separately from NVIDIA. Do not download CUDA DLLs from third-party sites.

## Verify GPU visibility

Open Command Prompt and run:

```bat
nvidia-smi
```

Then, from the LocalCaps folder after running `run.bat` once:

```bat
venv\Scripts\activate
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

A count greater than zero means CTranslate2 can see a CUDA device. LocalCaps also prints the selected runtime in the terminal at startup.

## Start

Double-click `run.bat`. It sets `WHISPER_DEVICE=auto`, so CUDA is selected when available. Otherwise the app safely runs on CPU. To force CPU, use `run-cpu.bat`.

## Important

GPU availability depends on your exact GPU, driver, CUDA and cuDNN installation. This ZIP has been prepared for automatic GPU detection and fallback, but GPU execution cannot be verified on a computer without an NVIDIA GPU and its CUDA libraries. The first use of each Whisper model downloads its model files; later runs use the local cache.
