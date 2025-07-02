import torch
from utils import mse, pair_downsampler

def denoise(model, noisy_img):
    """Denoise an image using the trained model"""
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
    return pred

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
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * torch.log10(1 / MSE)
    return PSNR
