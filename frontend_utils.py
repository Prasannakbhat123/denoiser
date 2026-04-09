from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import DenoisingNetwork
from training import calculate_mse_psnr, denoise_with_method
from utils import add_noise


ARTIFACTS_DIR = Path("artifacts")
BENCHMARK_JSON = ARTIFACTS_DIR / "benchmark_results.json"


def load_model(weights_path: str = "model_weights.pth") -> Tuple[Optional[DenoisingNetwork], torch.device, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingNetwork(n_chan=3).to(device)
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model, device, True
    except FileNotFoundError:
        return None, device, False


def load_benchmark_summary() -> List[Dict[str, object]]:
    if not BENCHMARK_JSON.exists():
        return []

    try:
        payload = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
        return payload.get("summary", [])
    except json.JSONDecodeError:
        return []


def benchmark_summary_frame() -> pd.DataFrame:
    rows = load_benchmark_summary()
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    preferred_columns = [
        "device",
        "method",
        "samples",
        "noise_type",
        "noise_level",
        "mean_mse",
        "mean_psnr_db",
        "latency_ms",
        "images_per_second",
        "speedup_vs_cpu",
    ]
    available = [column for column in preferred_columns if column in frame.columns]
    return frame[available]


def prepare_image(uploaded_file, max_size: int = 512):
    image = Image.open(uploaded_file).convert("RGB")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image, image_np, image_tensor


def add_noise_to_image(image_tensor: torch.Tensor, noise_level: int, noise_type: str = "gauss") -> torch.Tensor:
    return add_noise(image_tensor, noise_type=noise_type, noise_level=noise_level)


def run_denoising(
    noisy_img: torch.Tensor,
    method: str,
    model: Optional[DenoisingNetwork],
    kernel_size: int,
    sigma: float,
    passes: int = 1,
):
    current = noisy_img
    num_passes = max(1, int(passes))
    for _ in range(num_passes):
        current = denoise_with_method(
            current,
            method=method,
            model=model if method == "deep" else None,
            kernel_size=kernel_size,
            sigma=sigma,
            force_cuda=torch.cuda.is_available(),
        )
    return current


def compute_metrics(clean_img: torch.Tensor, noisy_img: torch.Tensor, denoised_img: torch.Tensor):
    noisy_mse, noisy_psnr = calculate_mse_psnr(clean_img, noisy_img)
    denoised_mse, denoised_psnr = calculate_mse_psnr(clean_img, denoised_img)
    return {
        "noisy_mse": noisy_mse,
        "noisy_psnr": noisy_psnr,
        "denoised_mse": denoised_mse,
        "denoised_psnr": denoised_psnr,
        "original_std": float(torch.std(clean_img)),
        "noisy_std": float(torch.std(noisy_img)),
        "denoised_std": float(torch.std(denoised_img)),
    }


def tensor_to_rgb_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def compare_all_methods(
    noisy_img: torch.Tensor,
    model: Optional[DenoisingNetwork],
    kernel_size: int,
    sigma: float,
    deep_passes: int = 1,
):
    methods = ["mean", "gaussian", "median"]
    if model is not None:
        methods = ["deep"] + methods

    outputs = {}
    for method in methods:
        method_passes = deep_passes if method == "deep" else 1
        outputs[method] = run_denoising(noisy_img, method, model, kernel_size, sigma, passes=method_passes)
    return outputs
