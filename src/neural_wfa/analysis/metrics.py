import torch
import numpy as np

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * torch.log10(mse).item()

def clamp_image(img: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """Clamps image values and rounds to 8-bit equivalent if precision is required."""
    return torch.clamp(img, min_val, max_val)

def model_size_in_bits(model: torch.nn.Module) -> int:
    """Calculates total number of bits used by model parameters and buffers."""
    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * p.element_size() * 8
    for b in model.buffers():
        total_bits += b.numel() * b.element_size() * 8
    return total_bits

def bits_per_pixel(model: torch.nn.Module, n_pixels: int) -> float:
    """Computes compression ratio in bits per pixel."""
    return model_size_in_bits(model) / n_pixels
