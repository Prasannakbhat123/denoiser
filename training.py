import torch
from utils import mse, pair_downsampler, mean_filter, gaussian_filter, median_filter

def denoise(model, noisy_img):
    """Denoise an image using the trained model"""
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
    return pred


def denoise_with_method(noisy_img, method='deep', model=None, kernel_size=3, sigma=1.0):
    method = method.lower()

    if method == 'deep':
        if model is None:
            raise ValueError("model is required when method='deep'")
        return denoise(model, noisy_img)
    if method == 'mean':
        return torch.clamp(mean_filter(noisy_img, kernel_size=kernel_size), 0, 1)
    if method == 'gaussian':
        return torch.clamp(gaussian_filter(noisy_img, kernel_size=kernel_size, sigma=sigma), 0, 1)
    if method == 'median':
        return torch.clamp(median_filter(noisy_img, kernel_size=kernel_size), 0, 1)

    raise ValueError(f"Unsupported denoising method: {method}")


def calculate_mse_psnr(clean_img, pred_img, eps=1e-12):
    mse_val = mse(clean_img, pred_img).item()
    mse_safe = max(mse_val, eps)
    psnr_val = 10.0 * torch.log10(torch.tensor(1.0 / mse_safe)).item()
    return mse_val, psnr_val

def loss_func(model, noisy_img):
    noisy1, noisy2 = pair_downsampler(noisy_img)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))
    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)
    loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
    return loss_res + loss_cons

def train(model, optimizer, noisy_img):
    loss = loss_func(model, noisy_img)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        _, PSNR = calculate_mse_psnr(clean_img, pred)
    return PSNR
