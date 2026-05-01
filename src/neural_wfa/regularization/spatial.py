import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_smoothing_kernel(device):
    """
    Returns the 3x3 smoothing kernel used in legacy code.
    """
    weights = torch.tensor(
        [[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]], device=device
    )
    weights = weights / weights.sum()
    return weights.view(1, 1, 3, 3)


def smoothness_loss(field_map: torch.Tensor, penalty: str = "l1") -> torch.Tensor:
    """
    Computes spatial smoothness loss (penalizing deviation from local mean).

    Args:
        field_map (torch.Tensor): 2D or 3D tensor (H, W) or (Batch, H, W) or (C, H, W).
                                  Will be treated as (Batch*C, 1, H, W) for convolution.
        penalty (str): 'l1' (Mean Absolute Diff) or 'l2' (Mean Squared Diff).

    Returns:
        torch.Tensor: Scalar loss.
    """
    if field_map.ndim == 2:
        # H, W -> 1, 1, H, W
        inp = field_map[None, None, :, :]
    elif field_map.ndim == 3:
        # C, H, W -> C, 1, H, W
        inp = field_map.unsqueeze(1)
    elif field_map.ndim == 4:
        # B, C, H, W -> B*C, 1, H, W
        B, C, H, W = field_map.shape
        inp = field_map.view(B * C, 1, H, W)
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got {field_map.ndim}D")

    weights = _get_smoothing_kernel(field_map.device)
    # Replicate weights if needed? No, grouped convolution or iterate?
    # view(B*C, 1, H, W) works with standard conv2d if weights is 1 channel out, 1 in.
    # We want to smooth each channel independently.

    # Pad to preserve size
    m = nn.ReflectionPad2d(1)
    padded = m(inp)

    smoothed = F.conv2d(padded, weights, padding="valid")

    diff = inp - smoothed

    if penalty == "l1":
        return torch.sum(torch.abs(diff))
    elif penalty == "l2":
        return torch.sum(diff**2)
    else:
        raise ValueError(f"Unknown penalty type: {penalty}")


def angle_smoothness_loss(angle_map: torch.Tensor, penalty: str = "l2") -> torch.Tensor:
    """
    Computes smoothness loss for angular variables (e.g. Phi) by enforcing
    continuity in 2*angle domain (sin(2x), cos(2x)).

    Legacy code uses sin(2x) and cos(2x).

    Args:
        angle_map (torch.Tensor): Tensor of angles (radians).

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Use 2*angle as in legacy regu2_angle
    sin_map = torch.sin(2 * angle_map)
    cos_map = torch.cos(2 * angle_map)

    loss_sin = smoothness_loss(sin_map, penalty=penalty)
    loss_cos = smoothness_loss(cos_map, penalty=penalty)

    return loss_sin + loss_cos


def mean_regularization(
    field_map: torch.Tensor, target_mean: float = None
) -> torch.Tensor:
    """
    Penalizes deviation of the entire field from its mean (or a target mean).

    Equivalent to regu_mean/regu_mean3 in legacy code.
    """
    if target_mean is None:
        target_mean = field_map.mean()
    return torch.sum(torch.abs(field_map - target_mean))


def min_value_regularization(field_map: torch.Tensor, min_val: float) -> torch.Tensor:
    """
    Penalizes values below a certain threshold.

    Equivalent to regu_min in legacy code.
    """
    return torch.sum(F.relu(-(field_map - min_val)))


def target_value_regularization(
    field_map: torch.Tensor, target_val: float
) -> torch.Tensor:
    """
    Penalizes deviation from a specific target value.

    Equivalent to regu_value in legacy code.
    """
    return torch.sum(torch.abs(field_map - target_val))


def potential_regularization(
    B_Q: torch.Tensor,
    B_U: torch.Tensor,
    target_Q: torch.Tensor,
    target_U: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizes deviation of the transverse magnetic field from a reference
    potential field.

    Args:
        B_Q (torch.Tensor): Predicted Stokes Q magnetic field component.
        B_U (torch.Tensor): Predicted Stokes U magnetic field component.
        target_Q (torch.Tensor): Reference potential Q field.
        target_U (torch.Tensor): Reference potential U field.

    Returns:
        torch.Tensor: Scalar loss (mean squared difference).
    """
    loss_Q = torch.mean((B_Q - target_Q) ** 2)
    loss_U = torch.mean((B_U - target_U) ** 2)
    return loss_Q + loss_U


def azimuth_regularization(
    B_Q: torch.Tensor, B_U: torch.Tensor, target_azimuth: torch.Tensor
) -> torch.Tensor:
    """
    Penalizes deviation of the magnetic field azimuth from a reference map
    (e.g., from fibrils), handling the 180-degree ambiguity.

    Uses the formulation:
        loss = sum((sin(2phi) - sin(2phi_ref))^2 + (cos(2phi) - cos(2phi_ref))^2)
    where phi is the azimuth angle derived from B_Q and B_U.

    Args:
        B_Q (torch.Tensor): Predicted Stokes Q magnetic field component.
        B_U (torch.Tensor): Predicted Stokes U magnetic field component.
        target_azimuth (torch.Tensor): Reference azimuth map in radians.

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Calculate predicted azimuth components (2*phi domain)
    # phi = 0.5 * atan2(U, Q)
    # We need sin(2phi) and cos(2phi).
    # sin(2phi) = U / sqrt(Q^2 + U^2)
    # cos(2phi) = Q / sqrt(Q^2 + U^2)

    Bt = torch.sqrt(B_Q**2 + B_U**2 + 1e-9)
    sin2phi_pred = B_U / Bt
    cos2phi_pred = B_Q / Bt

    sin2phi_ref = torch.sin(2 * target_azimuth)
    cos2phi_ref = torch.cos(2 * target_azimuth)

    loss_sin = torch.mean((sin2phi_pred - sin2phi_ref) ** 2)
    loss_cos = torch.mean((cos2phi_pred - cos2phi_ref) ** 2)

    return loss_sin + loss_cos
