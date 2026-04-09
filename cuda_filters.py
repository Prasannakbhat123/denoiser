from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

try:
    from numba import cuda
    _NUMBA_CUDA_IMPORT_OK = True
except Exception:
    cuda = None
    _NUMBA_CUDA_IMPORT_OK = False


def cuda_available() -> bool:
    return bool(_NUMBA_CUDA_IMPORT_OK and cuda is not None and cuda.is_available())


def _to_chw_float32(image_tensor: torch.Tensor) -> np.ndarray:
    tensor = image_tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return tensor.contiguous().numpy().astype(np.float32, copy=True)


def _to_tensor(image_chw: np.ndarray, reference_tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image_chw)).unsqueeze(0)
    return tensor.to(reference_tensor.device)


@cuda.jit
def _mean_filter_kernel(image_in, image_out, radius):
    x, y = cuda.grid(2)
    channels, height, width = image_in.shape

    if x >= width or y >= height:
        return

    for channel in range(channels):
        value_sum = 0.0
        sample_count = 0
        for dy in range(-radius, radius + 1):
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if nx < 0 or nx >= width:
                    continue
                value_sum += image_in[channel, ny, nx]
                sample_count += 1

        image_out[channel, y, x] = value_sum / sample_count


@cuda.jit
def _gaussian_filter_kernel(image_in, kernel, image_out, radius):
    x, y = cuda.grid(2)
    channels, height, width = image_in.shape

    if x >= width or y >= height:
        return

    for channel in range(channels):
        value_sum = 0.0
        for ky in range(-radius, radius + 1):
            ny = y + ky
            if ny < 0 or ny >= height:
                continue
            for kx in range(-radius, radius + 1):
                nx = x + kx
                if nx < 0 or nx >= width:
                    continue
                value_sum += image_in[channel, ny, nx] * kernel[ky + radius, kx + radius]

        image_out[channel, y, x] = value_sum


def _launch_2d_kernel(kernel, image_in, *kernel_args):
    height = image_in.shape[1]
    width = image_in.shape[2]
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](image_in, *kernel_args)
    cuda.synchronize()


def cuda_mean_filter(image_tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if not cuda_available():
        raise RuntimeError("CUDA is not available for mean filtering.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    radius = kernel_size // 2
    image_chw = _to_chw_float32(image_tensor)
    device_input = cuda.to_device(image_chw)
    device_output = cuda.device_array_like(device_input)

    _launch_2d_kernel(_mean_filter_kernel, device_input, device_output, radius)
    return _to_tensor(device_output.copy_to_host(), image_tensor)


def cuda_gaussian_filter(image_tensor: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    if not cuda_available():
        raise RuntimeError("CUDA is not available for gaussian filtering.")

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    coords = np.arange(kernel_size, dtype=np.float32) - radius
    yy, xx = np.meshgrid(coords, coords)
    kernel = np.exp(-((xx * xx) + (yy * yy)) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()

    image_chw = _to_chw_float32(image_tensor)
    device_input = cuda.to_device(image_chw)
    device_kernel = cuda.to_device(kernel)
    device_output = cuda.device_array_like(device_input)

    _launch_2d_kernel(_gaussian_filter_kernel, device_input, device_kernel, device_output, radius)
    return _to_tensor(device_output.copy_to_host(), image_tensor)