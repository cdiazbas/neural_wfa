import torch
import subprocess
import numpy as np

def get_free_gpu() -> torch.device:
    """
    Selects the GPU with the most free memory using nvidia-smi.
    """
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader"
        memory_free = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1]
        memory_free = [int(x) for x in memory_free]
        if not memory_free:
            return torch.device("cpu")
        best_gpu = np.argmax(memory_free)
        return torch.device(f"cuda:{best_gpu}")
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_scientific(value: float, precision: int = 2) -> str:
    """
    Formats a numerical value into a LaTeX-style scientific notation string.
    Example: 1234 -> 1.23 \times 10^3
    """
    if value == 0:
        return "0"
    mantissa, exp = f"{value:.{precision}e}".split("e")
    exp = int(exp)
    if exp == 0:
        return mantissa
    return f"{mantissa} \\times 10^{{{exp}}}"

def even_shape(x: np.ndarray) -> np.ndarray:
    """
    Crops an image array to ensure its dimensions are even.
    """
    ny, nx = x.shape[:2]
    new_ny = ny - (ny % 2)
    new_nx = nx - (nx % 2)
    return x[:new_ny, :new_nx, ...]

class AttributeDict(dict):
    """Dictionary subclass that allows attribute-style access."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def print_summary(n_pixels: int, n_params: int):
    """Prints a summary of pixels and parameters."""
    print(f"Number of parameters: {n_params}")
    print(f"Number of pixels: {n_pixels}")
    print(f"Pixels per parameter: {n_pixels / n_params:.2f}")
