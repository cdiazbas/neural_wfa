"""
Utility functions for probabilistic WFA.
"""

import torch
from typing import Union, Optional


def broadcast_sigma_obs(
    sigma_obs: Union[float, torch.Tensor],
    n_stokes: int = 3,
    n_wavelengths: int = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Broadcast observation noise to shape (3, Nw) for Q, U, V.
    
    Handles three granularity levels:
    - Global: scalar → broadcast to (3, Nw)
    - Per-Stokes: (3,) → broadcast to (3, Nw)
    - Full: (3, Nw) → use as-is
    
    Args:
        sigma_obs: Observation noise. Can be:
            - float or scalar tensor: Global noise
            - Tensor of shape (3,): Per-Stokes noise
            - Tensor of shape (3, Nw): Full specification
        n_stokes: Number of Stokes parameters (always 3: Q, U, V)
        n_wavelengths: Number of wavelength points
        device: Target device
        
    Returns:
        Tensor of shape (3, Nw) with broadcasted noise values
        
    Examples:
        >>> broadcast_sigma_obs(0.01, n_wavelengths=13)  # Global
        tensor([[0.01, 0.01, ...], ...])  # (3, 13)
        
        >>> broadcast_sigma_obs(torch.tensor([0.01, 0.02, 0.03]), n_wavelengths=13)  # Per-Stokes
        tensor([[0.01, ...], [0.02, ...], [0.03, ...]])  # (3, 13)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Convert to tensor if needed
    if isinstance(sigma_obs, (int, float)):
        sigma_obs = torch.tensor(sigma_obs, dtype=torch.float32, device=device)
    else:
        sigma_obs = sigma_obs.to(device)
    
    # Handle different input shapes
    if sigma_obs.ndim == 0:
        # Global: scalar → (3, Nw)
        if n_wavelengths is None:
            raise ValueError("n_wavelengths required for scalar sigma_obs")
        return sigma_obs.expand(n_stokes, n_wavelengths).clone()
    
    elif sigma_obs.ndim == 1:
        if sigma_obs.shape[0] == 1:
            # Global: (1,) → (3, Nw)
            if n_wavelengths is None:
                raise ValueError("n_wavelengths required for global sigma_obs")
            return sigma_obs.expand(n_stokes, n_wavelengths).clone()
        elif sigma_obs.shape[0] == n_stokes:
            # Per-Stokes: (3,) → (3, Nw)
            if n_wavelengths is None:
                raise ValueError("n_wavelengths required for 1D sigma_obs")
            return sigma_obs.unsqueeze(1).expand(n_stokes, n_wavelengths).clone()
        else:
            raise ValueError(f"Expected 1 (global) or {n_stokes} (per-stokes) values, got {sigma_obs.shape[0]}")
    
    elif sigma_obs.ndim == 2:
        # Full: (3, Nw) → use as-is
        if sigma_obs.shape[0] != n_stokes:
            raise ValueError(f"Expected first dim {n_stokes}, got {sigma_obs.shape[0]}")
        if n_wavelengths is not None and sigma_obs.shape[1] != n_wavelengths:
            raise ValueError(f"Expected {n_wavelengths} wavelengths, got {sigma_obs.shape[1]}")
        return sigma_obs.clone()
    
    else:
        raise ValueError(f"sigma_obs must be scalar, 1D, or 2D. Got shape {sigma_obs.shape}")


def create_learnable_sigma_obs(
    granularity: str = 'per_stokes',
    n_stokes: int = 3,
    n_wavelengths: int = None,
    init_value: float = 0.01,
    device: torch.device = None,
) -> torch.nn.Parameter:
    """
    Create a learnable log(sigma_obs) parameter.
    
    Args:
        granularity: 'global', 'per_stokes', or 'full'
        n_stokes: Number of Stokes parameters (3)
        n_wavelengths: Number of wavelength points
        init_value: Initial sigma value (will be log-transformed)
        device: Target device
        
    Returns:
        nn.Parameter with log(sigma_obs) values
    """
    init_log = torch.log(torch.tensor(init_value))
    
    if granularity == 'global':
        shape = (1,)
    elif granularity == 'per_stokes':
        shape = (n_stokes,)
    elif granularity == 'full':
        if n_wavelengths is None:
            raise ValueError("n_wavelengths required for 'full' granularity")
        shape = (n_stokes, n_wavelengths)
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    
    log_sigma = torch.full(shape, init_log, device=device)
    return torch.nn.Parameter(log_sigma)


def log_sigma_to_sigma(
    log_sigma: torch.Tensor,
    n_stokes: int = 3,
    n_wavelengths: int = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert learnable log(sigma) to broadcasted sigma (3, Nw).
    
    Args:
        log_sigma: Log of sigma values (any granularity)
        n_stokes: Number of Stokes (3)
        n_wavelengths: Number of wavelengths
        device: Target device
        
    Returns:
        Tensor of shape (3, Nw)
    """
    sigma = torch.exp(log_sigma)
    return broadcast_sigma_obs(sigma, n_stokes, n_wavelengths, device)
