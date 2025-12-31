import torch
import torch.nn.functional as F

def temporal_smoothness_loss(params_time_series: torch.Tensor) -> torch.Tensor:
    """
    Computes time-domain regularization using squared differences between neighbors.
    
    Args:
        params_time_series: Shape (n_pixels, n_time) or (n_pixels, n_time, n_params).
        
    Returns:
        Scalar loss.
    """
    # torch.diff computes x[t+1] - x[t]
    diffs = torch.diff(params_time_series, dim=1)
    return torch.sum(diffs**2)

def temporal_total_variation(params_time_series: torch.Tensor) -> torch.Tensor:
    """
    Computes time-domain regularization using absolute differences (L1).
    """
    diffs = torch.diff(params_time_series, dim=1)
    return torch.sum(torch.abs(diffs))
