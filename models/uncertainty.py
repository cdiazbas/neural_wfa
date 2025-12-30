import torch
import numpy as np

def estimate_uncertainties_analytical(model, params, index=None):
    """
    Analytical uncertainty estimation using error propagation.
    """
    # Inline estimate_uncertainties_internal logic
    if index is None:
        index = slice(None) # Use all indices
    
    # Calculate Residuals Sum of Squares (Chi2)
    stokesQ, stokesU, stokesV = model.forward(params, index=index)
    
    # Compute Chi-squared per pixel
    # Note: stokes* already corresponds to index if index was passed to forward
    # But self.data_stokes* needs indexing
    if isinstance(index, slice) and index == slice(None):
        # Full dataset
        idx_op = slice(None)
    else:
        idx_op = index
        
    chi2_Q = torch.sum((model.data_stokesQ[idx_op, :] - stokesQ)[:, model.mask] ** 2.0, dim=1)
    chi2_U = torch.sum((model.data_stokesU[idx_op, :] - stokesU)[:, model.mask] ** 2.0, dim=1)
    chi2_V = torch.sum((model.data_stokesV[idx_op, :] - stokesV)[:, model.mask] ** 2.0, dim=1)
    
    n_wavelengths = len(model.mask)
    
    # Calculate uncertainties
    # Denominators
    Blos_denom = n_wavelengths * abs(model.C * model.lin.geff)**2 * torch.sum(model.dIdw[idx_op, :][:, model.mask] ** 2, dim=-1)
    Blos_uncertainty = torch.sqrt(chi2_V / (Blos_denom + 1e-20))
    
    denom_Q = n_wavelengths * (0.75 * model.C**2 * model.lin.Gg)**2 * torch.sum(model.dIdwscl[idx_op, :][:, model.mask] ** 2, dim=-1)
    BQ_uncertainty = torch.sqrt(chi2_Q / (denom_Q + 1e-20))
    
    BU_uncertainty = torch.sqrt(
        chi2_U / (n_wavelengths * (0.75 * model.C**2 * model.lin.Gg)**2 * torch.sum(model.dIdwscl[idx_op, :][:, model.mask] ** 2, dim=-1) + 1e-20)
    )
    
    # Extract BQ and BU from params (normalized)
    BQ = params[:, 1] * model.QUnorm
    BU = params[:, 2] * model.QUnorm
    
    # Calculate Btr_sq = BQ^2 + BU^2
    Btr_sq = BQ**2 + BU**2
    
    # Calculate the uncertainties using error propagation
    dBtr_dQ = BQ / (2.0 * Btr_sq**0.75 + 1e-9)
    dBtr_dU = BU / (2.0 * Btr_sq**0.75 + 1e-9)
    
    uncertainty_btr = torch.sqrt((dBtr_dQ**2 * BQ_uncertainty**2) + (dBtr_dU**2 * BU_uncertainty**2))

    # For phiB
    dphiB_dQ = -0.5 * BU / (BQ**2 + BU**2 + 1e-9)
    dphiB_dU = 0.5 * BQ / (BQ**2 + BU**2 + 1e-9)

    uncertainty_phib = torch.sqrt((dphiB_dQ**2 * BQ_uncertainty**2) + (dphiB_dU**2 * BU_uncertainty**2))

    # Reshape the output
    n_total = model.ny * model.nx * model.nt
    if len(Blos_uncertainty) == n_total:
        uncertainty_blos = Blos_uncertainty.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        uncertainty_btr = uncertainty_btr.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        uncertainty_phib = uncertainty_phib.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
    else:
        # Handle subset
        full_blos = torch.full((n_total,), float('nan'), device=Blos_uncertainty.device)
        full_btr = torch.full((n_total,), float('nan'), device=Blos_uncertainty.device)
        full_phi = torch.full((n_total,), float('nan'), device=Blos_uncertainty.device)
        
        if isinstance(index, range):
            index = list(index)
        full_blos[idx_op] = Blos_uncertainty
        full_btr[idx_op] = uncertainty_btr
        full_phi[idx_op] = uncertainty_phib
        
        uncertainty_blos = full_blos.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        uncertainty_btr = full_btr.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        uncertainty_phib = full_phi.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()

    return uncertainty_blos, uncertainty_btr, uncertainty_phib


def estimate_uncertainties_taylor(model, params, index=None):
    """
    Estimate uncertainties using explicit Taylor expansion.
    """
    stokesQ, stokesU, stokesV = model.forward(params, index=index)
    
    if index is None:
        index = slice(None)
    
    if isinstance(index, slice) and index == slice(None):
        idx_op = slice(None)
    else:
        idx_op = index
        
    residuals_V_sq = (model.data_stokesV[idx_op, :] - stokesV)[:, model.mask] ** 2.0
    residuals_Q_sq = (model.data_stokesQ[idx_op, :] - stokesQ)[:, model.mask] ** 2.0
    residuals_U_sq = (model.data_stokesU[idx_op, :] - stokesU)[:, model.mask] ** 2.0
    
    rms_V = torch.sqrt(torch.mean(residuals_V_sq, dim=1))
    rms_Q = torch.sqrt(torch.mean(residuals_Q_sq, dim=1))
    rms_U = torch.sqrt(torch.mean(residuals_U_sq, dim=1))
    
    dIdw_sq = model.dIdw[idx_op, :][:, model.mask] ** 2
    sensitivity_Blos = abs(model.C * model.lin.geff) * torch.sqrt(torch.sum(dIdw_sq, dim=1))
    sigma_Blos = rms_V / (sensitivity_Blos + 1e-9)
    
    dIdwscl_sq = model.dIdwscl[idx_op, :][:, model.mask] ** 2
    sensitivity_BQ_BU = abs(0.75 * model.C**2 * model.lin.Gg) * torch.sqrt(torch.sum(dIdwscl_sq, dim=1))
    
    sigma_BQ = rms_Q / (sensitivity_BQ_BU + 1e-12)
    sigma_BU = rms_U / (sensitivity_BQ_BU + 1e-12)
    
    BQ = params[:, 1] * model.QUnorm
    BU = params[:, 2] * model.QUnorm
    Btr_sq = BQ**2 + BU**2
    
    dBt_dBQ = BQ / (2.0 * Btr_sq**0.75 + 1e-9)
    dBt_dBU = BU / (2.0 * Btr_sq**0.75 + 1e-9)
    
    sigma_Bt = torch.sqrt((dBt_dBQ**2 * sigma_BQ**2) + (dBt_dBU**2 * sigma_BU**2))
    
    dphi_dBQ = -0.5 * BU / (Btr_sq + 1e-9)
    dphi_dBU = 0.5 * BQ / (Btr_sq + 1e-9)
    
    sigma_phi = torch.sqrt((dphi_dBQ**2 * sigma_BQ**2) + (dphi_dBU**2 * sigma_BU**2))
    
    n_total = model.ny * model.nx * model.nt
    if len(sigma_Blos) == n_total:
        return (
            sigma_Blos.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            sigma_Bt.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            sigma_phi.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        )
    else:
        full_blos = torch.full((n_total,), float('nan'), device=sigma_Blos.device)
        full_bt = torch.full((n_total,), float('nan'), device=sigma_Blos.device)
        full_phi = torch.full((n_total,), float('nan'), device=sigma_Blos.device)
        
        if isinstance(index, range):
            index = list(index)
        full_blos[idx_op] = sigma_Blos
        full_bt[idx_op] = sigma_Bt
        full_phi[idx_op] = sigma_phi
        
        return (
            full_blos.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            full_bt.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            full_phi.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        )


def estimate_uncertainties_pytorch(model, params, index=None):
    """
    PyTorch-derived uncertainty estimation using vectorized Hessian.
    """
    if not params.requires_grad:
        params = params.clone().detach().requires_grad_(True)
    
    # Forward pass logic is handled within the loss closure for Hessians
    # But we need data corresponding to params
    
    if index is None:
        idx_op = slice(None)
    else:
        idx_op = index
    
    # Prepare batch data
    batch_Q = model.data_stokesQ[idx_op, :]
    batch_U = model.data_stokesU[idx_op, :]
    batch_V = model.data_stokesV[idx_op, :]
    
    batch_dIdw = model.dIdw[idx_op, :]
    batch_dIdwscl = model.dIdwscl[idx_op, :]
    
    # Capture model variables
    C = model.C
    geff = model.lin.geff
    Gg = model.lin.Gg
    Vnorm = model.Vnorm
    QUnorm = model.QUnorm
    mask = model.mask
    
    # Define single-pixel loss function
    def single_pixel_loss(p, dQ, dU, dV, dIdw, dIdwscl):
        # p: (3,)
        Blos_norm, BQ_norm, BU_norm = p[0], p[1], p[2]
        
        # Forward model
        stokesV = C * geff * (Blos_norm * Vnorm) * dIdw
        
        Clp = 0.75 * C**2 * Gg * dIdwscl
        stokesQ = Clp * (BQ_norm * QUnorm)
        stokesU = Clp * (BU_norm * QUnorm)
        
        # Residuals (masked)
        res_Q = (dQ - stokesQ)[mask]
        res_U = (dU - stokesU)[mask]
        res_V = (dV - stokesV)[mask]
        
        return torch.sum(res_Q**2 + res_U**2 + res_V**2)

    try:
        from torch.func import hessian, vmap
    except ImportError:
        print("Warning: torch.func not found.")
        raise ImportError("PyTorch >= 2.0 with torch.func is required.")

    # Vectorized Hessian
    batch_hessian_fn = vmap(hessian(single_pixel_loss), in_dims=(0, 0, 0, 0, 0, 0))
    hessians = batch_hessian_fn(params, batch_Q, batch_U, batch_V, batch_dIdw, batch_dIdwscl)
    
    # Regularize and Invert
    hessians = 0.5 * (hessians + hessians.transpose(-2, -1))
    hessians = hessians + 1e-6 * torch.eye(3, device=params.device).unsqueeze(0)
    
    covariances = 2.0 * torch.linalg.inv(hessians)
    variances = torch.diagonal(covariances, dim1=-2, dim2=-1)
    
    sigma_Blos_norm = torch.sqrt(torch.abs(variances[:, 0]))
    sigma_BQ_norm = torch.sqrt(torch.abs(variances[:, 1]))
    sigma_BU_norm = torch.sqrt(torch.abs(variances[:, 2]))
    
    Blos_uncertainty = sigma_Blos_norm * abs(model.Vnorm)
    BQ_uncertainty = sigma_BQ_norm * abs(model.QUnorm)
    BU_uncertainty = sigma_BU_norm * abs(model.QUnorm)
    
    # Propagate
    BQ = params[:, 1] * model.QUnorm
    BU = params[:, 2] * model.QUnorm
    
    Btr_sq = BQ**2 + BU**2
    
    dBtr_dQ = BQ / (2.0 * Btr_sq**0.75 + 1e-9)
    dBtr_dU = BU / (2.0 * Btr_sq**0.75 + 1e-9)
    
    uncertainty_btr = torch.sqrt((dBtr_dQ**2 * BQ_uncertainty**2) + (dBtr_dU**2 * BU_uncertainty**2))
    
    dphiB_dQ = -0.5 * BU / (BQ**2 + BU**2 + 1e-9)
    dphiB_dU = 0.5 * BQ / (BQ**2 + BU**2 + 1e-9)
    
    uncertainty_phib = torch.sqrt((dphiB_dQ**2 * BQ_uncertainty**2) + (dphiB_dU**2 * BU_uncertainty**2))
    
    # Reshape
    n_total = model.ny * model.nx * model.nt
    if len(Blos_uncertainty) == n_total:
        return (
            Blos_uncertainty.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            uncertainty_btr.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            uncertainty_phib.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        )
    else:
        full_blos = torch.full((n_total,), float('nan'), device=params.device)
        full_bt = torch.full((n_total,), float('nan'), device=params.device)
        full_phi = torch.full((n_total,), float('nan'), device=params.device)
        
        if isinstance(index, range):
            index = list(index)
            
        full_blos[idx_op] = Blos_uncertainty
        full_bt[idx_op] = uncertainty_btr
        full_phi[idx_op] = uncertainty_phib
        
        return (
            full_blos.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            full_bt.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy(),
            full_phi.reshape(model.ny, model.nx, model.nt).detach().cpu().numpy()
        )
