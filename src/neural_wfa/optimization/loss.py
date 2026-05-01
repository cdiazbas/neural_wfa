import torch
import torch.nn.functional as F


def huber_loss(
    input: torch.Tensor, target: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """
    Computes the Huber loss between input and target.

    Acts like L2 loss for small residuals and L1 loss for large
    outliers.
    """
    return F.huber_loss(input, target, delta=delta)


def cauchy_loss(
    input: torch.Tensor, target: torch.Tensor, c: float = 1.0
) -> torch.Tensor:
    """
    Computes the Cauchy loss (Log-Cosh like) between input and target.

    More robust to extreme outliers than Huber loss.
    """
    residual = input - target
    loss = torch.log(1 + (residual / c) ** 2)
    return torch.mean(loss)
