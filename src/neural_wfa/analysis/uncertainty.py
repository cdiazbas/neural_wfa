import torch
import numpy as np
from typing import Dict, Tuple, Optional
from neural_wfa.core.problem import WFAProblem
from neural_wfa.core.magnetic_field import MagneticField

def estimate_uncertainties_diagonal(
    problem: WFAProblem, 
    field: MagneticField, 
    indices: torch.Tensor = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate uncertainties using the diagonal approximation of the Hessian (Analytical Error Propagation).
    Matches legacy 'analytical' and 'taylor' logic.
    """
    stokesQ, stokesU, stokesV = problem.compute_forward_model(field, indices=indices)
    
    if indices is None:
        idx_op = slice(None)
        dIdw = problem.dIdw
        dIdwscl = problem.dIdwscl
    else:
        idx_op = indices
        dIdw = problem.dIdw[indices]
        dIdwscl = problem.dIdwscl[indices]

    mask = problem.mask
    obs_Q = problem.obs.stokes_Q[idx_op]
    obs_U = problem.obs.stokes_U[idx_op]
    obs_V = problem.obs.stokes_V[idx_op]

    # Handle None mask (Fix: Must be done before residuals calculation)
    if mask is None:
        n_lambda = obs_Q.shape[-1]
        mask = slice(None)
    else:
        n_lambda = len(mask)

    # Handle time dimension if present
    if stokesV.ndim == 3:
        # Model is (N, T, L). Flatten or repeat Obs?
        # Usually uncertainties are per time step.
        # For now, let's process assuming T=1 or flattened.
        pass

    # Residuals Sum of Squares
    res_V_sq = torch.sum((obs_V - stokesV.squeeze(1))[..., mask]**2, dim=-1)
    res_Q_sq = torch.sum((obs_Q - stokesQ.squeeze(1))[..., mask]**2, dim=-1)
    res_U_sq = torch.sum((obs_U - stokesU.squeeze(1))[..., mask]**2, dim=-1)
    
    # Sensitivities
    sens_blos = abs(problem.C * problem.lin.geff) * torch.sqrt(torch.sum(dIdw[..., mask]**2, dim=-1))
    sigma_blos = torch.sqrt(res_V_sq / (n_lambda * sens_blos**2 + 1e-20))
    
    sens_bqu = abs(0.75 * problem.C**2 * problem.lin.Gg) * torch.sqrt(torch.sum(dIdwscl[..., mask]**2, dim=-1))
    sigma_bq = torch.sqrt(res_Q_sq / (n_lambda * sens_bqu**2 + 1e-20))
    sigma_bu = torch.sqrt(res_U_sq / (n_lambda * sens_bqu**2 + 1e-20))
    
    # Propagation
    BQ = field.b_q
    if BQ.dim() > 1 and BQ.shape[1] == 1:
        BQ = BQ.squeeze(1)
    BU = field.b_u
    if BU.dim() > 1 and BU.shape[1] == 1:
        BU = BU.squeeze(1)
    Btr_sq = BQ**2 + BU**2
    Btr = torch.sqrt(Btr_sq)
    
    # Btr here is B_perp^2. 
    # sigma_btrans_sq = sigma(B_perp^2)
    # sigma_btrans_lin = sigma(B_perp^2) / (2 * B_perp) = sigma_sq / (2 * sqrt(Btr))
    sigma_btrans_sq = torch.sqrt( ( (BQ/(Btr+1e-9))*sigma_bq )**2 + ( (BU/(Btr+1e-9))*sigma_bu )**2 )
    sigma_btrans = sigma_btrans_sq / (2 * torch.sqrt(Btr + 1e-9) + 1e-9)
    sigma_phi = 0.5 * torch.sqrt( ( (BU/(Btr_sq+1e-9))*sigma_bq )**2 + ( (BQ/(Btr_sq+1e-9))*sigma_bu )**2 )
    
    return (
        sigma_blos.detach().cpu().numpy(),
        sigma_btrans.detach().cpu().numpy(),
        sigma_phi.detach().cpu().numpy()
    )

def estimate_uncertainties(
    problem: WFAProblem,
    field: MagneticField,
    method: str = 'analytical',
    indices: torch.Tensor = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified entry point for uncertainty estimation.
    """
    if method in ['analytical', 'taylor', 'diagonal']:
        return estimate_uncertainties_diagonal(problem, field, indices=indices)
    elif method in ['pytorch', 'hessian']:
        return estimate_uncertainties_pytorch(problem, field, indices=indices)
    else:
        raise ValueError(f"Unknown method: {method}")

def estimate_uncertainties_pytorch(
    problem: WFAProblem,
    field: MagneticField,
    indices: torch.Tensor = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate uncertainties using vectorized Hessian calculation via torch.func.
    """
    from torch.func import hessian, vmap
    
    if indices is None:
        idx_op = slice(None)
        batch_dIdw = problem.dIdw
        batch_dIdwscl = problem.dIdwscl
    else:
        idx_op = indices
        batch_dIdw = problem.dIdw[indices]
        batch_dIdwscl = problem.dIdwscl[indices]
        
    batch_Q = problem.obs.stokes_Q[idx_op]
    batch_U = problem.obs.stokes_U[idx_op]
    batch_V = problem.obs.stokes_V[idx_op]
    
    # Constant params for closure
    C = problem.C
    geff = problem.lin.geff
    Gg = problem.lin.Gg
    mask = problem.mask

    def pixel_loss(p_norm, dQ, dU, dV, dIdw, dIdwscl):
        # p_norm: (3,) [Blos, BQ, BU] normalized values
        # We assume field weights are 1.0 for simplicity in Hessian context
        sv = C * geff * p_norm[0] * dIdw
        clp = 0.75 * C**2 * Gg * dIdwscl
        sq = clp * p_norm[1]
        su = clp * p_norm[2]
        return torch.sum((dQ[mask]-sq[mask])**2 + (dU[mask]-su[mask])**2 + (dV[mask]-sv[mask])**2)

    # Get current normalized params from field
    # (N, 3)
    p_init = torch.stack([field.blos, field.b_q, field.b_u], dim=-1).squeeze(1).detach().requires_grad_(True)
    
    batch_hessian_fn = vmap(hessian(pixel_loss), in_dims=(0, 0, 0, 0, 0, 0))
    hessians = batch_hessian_fn(p_init, batch_Q, batch_U, batch_V, batch_dIdw, batch_dIdwscl)
    
    # Invert to get Variances
    hessians = 0.5 * (hessians + hessians.transpose(-1, -2)) + 1e-8 * torch.eye(3, device=p_init.device)
    covariances = 2.0 * torch.linalg.inv(hessians)
    variances = torch.diagonal(covariances, dim1=-1, dim2=-2)
    
    # Normalized sigmas
    sigma_p = torch.sqrt(torch.abs(variances))
    
    # Back to physical and propagate
    sigma_blos = sigma_p[:, 0]
    sigma_bq = sigma_p[:, 1]
    sigma_bu = sigma_p[:, 2]
    
    BQ = field.b_q
    if BQ.dim() > 1 and BQ.shape[1] == 1:
        BQ = BQ.squeeze(1)
    BU = field.b_u
    if BU.dim() > 1 and BU.shape[1] == 1:
        BU = BU.squeeze(1)
    Btr_sq = BQ**2 + BU**2
    Btr = torch.sqrt(Btr_sq)
    
    # Btr here is B_perp^2. 
    # sigma_btrans_sq = sigma(B_perp^2)
    # sigma_btrans_lin = sigma(B_perp^2) / (2 * B_perp) = sigma_sq / (2 * sqrt(Btr))
    sigma_btrans_sq = torch.sqrt( ( (BQ/(Btr+1e-9))*sigma_bq )**2 + ( (BU/(Btr+1e-9))*sigma_bu )**2 )
    sigma_btrans = sigma_btrans_sq / (2 * torch.sqrt(Btr + 1e-9) + 1e-9)
    sigma_phi = 0.5 * torch.sqrt( ( (BU/(Btr_sq+1e-9))*sigma_bq )**2 + ( (BQ/(Btr_sq+1e-9))*sigma_bu )**2 )
    
    return (
        sigma_blos.detach().cpu().numpy(),
        sigma_btrans.detach().cpu().numpy(),
        sigma_phi.detach().cpu().numpy()
    )
