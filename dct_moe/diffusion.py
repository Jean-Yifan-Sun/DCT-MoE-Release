"""Continuous VP diffusion and the original SNR-adjusted DPM-Solver."""
import torch
from .dpm_solver import DPM_Solver, NoiseScheduleVP


def diffusion_loss(model, clean, snr_scale=4., entropy=None):
    schedule = NoiseScheduleVP(SNR_scale=snr_scale)
    t = torch.rand(clean.shape[0], device=clean.device)
    log_alpha = schedule.marginal_log_mean_coeff(t)
    alpha = log_alpha.exp()[:, None, None]
    sigma = (-torch.expm1(2 * log_alpha)).sqrt()[:, None, None]
    noise = torch.randn_like(clean)
    noisy = alpha * clean + sigma * noise
    prediction, aux = model(noisy, t * 999)
    squared_error = (prediction.float() - noise.float()).square()
    if entropy is not None:
        weights = torch.as_tensor(entropy, device=clean.device, dtype=torch.float32).repeat(4)
        if weights.numel() != clean.shape[-1] or not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('entropy must contain one finite positive value per retained frequency')
        weights = weights / weights.mean()
        squared_error = squared_error * weights
    return squared_error.mean(), aux


@torch.no_grad()
def sample_tokens(model, noise, snr_scale=4., steps=50):
    if steps < 1:
        raise ValueError('steps (NFE) must be positive')
    if model.training:
        raise ValueError('Call model.eval() before deterministic ODE sampling')
    schedule = NoiseScheduleVP(SNR_scale=snr_scale)
    def predict(x, t):
        return model(x, t * 999)[0]
    return DPM_Solver(predict, schedule).sample(
        noise, steps=steps, eps=1e-4, adaptive_step_size=False, fast_version=True)
