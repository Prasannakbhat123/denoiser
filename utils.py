import torch
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_noise(x, noise_type='gauss', noise_level=25):
    if noise_type == 'gauss':
        noisy = x + torch.randn_like(x) * (noise_level / 255.0)
        noisy = torch.clamp(noisy, 0, 1)
    elif noise_type == 'poiss':
        noisy = torch.poisson(noise_level * x) / noise_level
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    return noisy


def _make_depthwise_kernel(base_kernel, channels, device, dtype):
    return base_kernel.to(device=device, dtype=dtype).repeat(channels, 1, 1, 1)


def mean_filter(img, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    channels = img.shape[1]
    base = torch.ones((1, 1, kernel_size, kernel_size), device=img.device, dtype=img.dtype)
    base = base / (kernel_size * kernel_size)
    kernel = _make_depthwise_kernel(base, channels, img.device, img.dtype)
    return F.conv2d(img, kernel, padding=kernel_size // 2, groups=channels)


def gaussian_filter(img, kernel_size=5, sigma=1.0):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    coords = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - (kernel_size - 1) / 2.0
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel_2d = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    kernel_2d = kernel_2d / kernel_2d.sum()

    base = kernel_2d.view(1, 1, kernel_size, kernel_size)
    channels = img.shape[1]
    kernel = _make_depthwise_kernel(base, channels, img.device, img.dtype)
    return F.conv2d(img, kernel, padding=kernel_size // 2, groups=channels)


def median_filter(img, kernel_size=3):
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    pad = kernel_size // 2
    padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
    b, c, h, w = img.shape

    unfolded = F.unfold(padded, kernel_size=kernel_size)
    unfolded = unfolded.view(b, c, kernel_size * kernel_size, h * w)
    median_vals = unfolded.median(dim=2).values
    return median_vals.view(b, c, h, w)

def pair_downsampler(img):
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device).repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device).repeat(c, 1, 1, 1)
    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2

def mse(gt, pred):
    loss = torch.nn.MSELoss()
    return loss(gt, pred)
