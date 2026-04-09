# 🖼️ CUDA-Accelerated Image Denoising

This project implements classical image denoising, a deep denoising model, and CUDA-backed filtering so you can compare CPU and GPU execution times and quality metrics.

## What This Project Does

- Real-time image denoising using deep learning
- Classical mean, Gaussian, and median filtering
- CUDA-backed mean and Gaussian filtering when an NVIDIA GPU is available
- CPU vs GPU benchmarking with execution time, speedup, and time-savings percentage
- Interactive Streamlit interface for image upload and result comparison

## Files

- [denoisingapp.py](denoisingapp.py) - Streamlit frontend
- [train_model.py](train_model.py) - Training and noisy-dataset generation
- [benchmark.py](benchmark.py) - CPU/GPU benchmark runner
- [training.py](training.py) - Denoising dispatcher and loss functions
- [cuda_filters.py](cuda_filters.py) - Numba CUDA kernels for classical filters
- [run_project.bat](run_project.bat) - One-click Windows launcher

## Quick Start on Windows

Double-click [run_project.bat](run_project.bat) or run:

```bat
run_project.bat
```

The batch file will:

1. Create `.venv` if it does not exist
2. Activate the environment
3. Upgrade `pip`
4. Install CUDA-enabled PyTorch wheels when `nvidia-smi` is available, otherwise CPU wheels
5. Install the remaining packages from [requirements.txt](requirements.txt)
6. Start the Streamlit app

## Benchmarking CPU vs GPU

To measure execution time for multiple image sizes and compare CPU vs CUDA:

```bat
run_project.bat benchmark
```

This runs the benchmark for sizes `64,128,256` and writes results to `artifacts_benchmark/`.

The benchmark outputs include:

- `latency_ms`
- `images_per_second`
- `speedup_vs_cpu`
- `time_savings_pct`
- `mean_mse`
- `mean_psnr_db`

To compare a different set of sizes, edit the `--sizes` argument in [run_project.bat](run_project.bat) or run [benchmark.py](benchmark.py) directly.

## Training on Clean and Noisy Pairs

To generate a noisy copy dataset and train from paired images:

```bat
run_project.bat train
```

This creates `data_noisy/` and trains a model using paired clean/noisy images.

## Notes on CUDA

- If CUDA is available, PyTorch uses the GPU for the deep model.
- Mean and Gaussian classical filters use explicit Numba CUDA kernels when GPU support is available.
- If no CUDA device is found, the project falls back to CPU automatically.

## Report Alignment

This implementation supports the report sections for:

- Introduction
- Literature Review
- Problem Statement
- Objectives
- Methodology
- Experimental Setup
- Results and Analysis
- Conclusions and Future Work

## Troubleshooting

- If the app starts on a different port, Streamlit may already be running elsewhere.
- If CUDA is not used, check that `torch.cuda.is_available()` returns `True` on the target PC.
- If `numba` or CUDA kernels fail, the app still works on CPU.

## Dependencies

Main packages:

- `streamlit`
- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `numba`

---

If you want a fully CUDA-kernel-based version for more operators later, the project is now set up so it can grow in that direction cleanly.