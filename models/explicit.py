import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys
from tqdm import tqdm, trange
import astropy.io.fits as fits


# =================================================================
def spatial_regularization(out_params_flat, param_idx, ny, nx):
    """
    The regularization is implemented as a convolution with a 3x3 kernel.
    Applies to a single parameter (param_idx) across spatial dimensions (ny, nx)
    for a single time step.

    Args:
        out_params_flat (torch.Tensor): Tensor of shape (ny * nx, n_model_params)
                                        containing parameters for a single time step.
        param_idx (int): Index of the parameter to regularize (0 for Blos, etc.).
        ny (int): Number of pixels in y-direction.
        nx (int): Number of pixels in x-direction.

    Returns:
        torch.Tensor: Scalar regularization loss for this parameter and time step.
    """
    # Extract the specific parameter and reshape to (1, 1, ny, nx) for conv2d
    param_2d = out_params_flat[:, param_idx].reshape(1, 1, ny, nx)

    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights / weights.sum()
    weights = weights.view(1, 1, 3, 3) # Batch, In_channels, Height, Width

    # Apply convolution
    output_smooth = F.conv2d(m(param_2d), weights, padding='valid')

    # Squared difference between the original and the neighbor pixels
    return torch.sum(torch.abs(output_smooth - param_2d)**2.0)


# =================================================================
def temporal_regularization(params_time_series):
    """
    Computes the time-domain regularization.
    Args:
        params_time_series (torch.Tensor): Tensor of shape (n_pixels, n_time) for a single parameter.
    Returns:
        torch.Tensor: The scalar regularization loss.
    """
    # Calculate difference between consecutive time steps
    # We penalize the squared difference: (param[t] - param[t-1])^2
    # torch.diff computes the difference along the last dimension.
    # It will result in a tensor of shape (n_pixels, n_time - 1)
    diffs = torch.diff(params_time_series, dim=-1)
    return torch.sum(torch.abs(diffs)**2.0)


# =================================================================
def optimization(optimizer, niterations, parameters, model, 
                 reguV=1e-3, reguQU=0.5e-1, reguT_Blos=1e-3, reguT_Bhor=1e-3, reguT_Bazi=1e-3, 
                 weights=[10,10,1], normgrad=False, mask=None):
    # Optimization loop
    
    # Explicit WFA
    model.Vnorm = 1000.0
    model.QUnorm = 1e6
    
    # parameters shape: (ny * nx, nTime, n_model_params)

    t = trange(niterations, leave=True)
    for loop in t:
        optimizer.zero_grad()

        # Chi2 loss (fidelity to data)
        chi2loss = model.evaluate(parameters, weights=weights, spatial_mask=mask)

        # Reshape parameters for easier spatial/temporal regularization access
        # (ny * nx, nTime, n_model_params)

        # If there are multiple time steps, we reshape to (ny * nx, nt, n_model_params)
        # This allows us to access each time step's parameters easily
        params_reshaped = parameters.reshape(model.ny * model.nx, model.nt, -1)

        # --- Spatial regularization ---
        total_spatial_loss = torch.tensor(0.0, device=parameters.device)
        for t_idx in range(model.nt):
            # Extract parameters for the current time step (ny*nx, n_model_params)
            current_time_params = params_reshaped[:, t_idx, :]
            
            if reguV > 0:
                total_spatial_loss += reguV * spatial_regularization(current_time_params, 0, model.ny, model.nx) # Blos
            if reguQU > 0:
                total_spatial_loss += reguQU * spatial_regularization(current_time_params, 1, model.ny, model.nx) # BQ (from Bt)
                total_spatial_loss += reguQU * spatial_regularization(current_time_params, 2, model.ny, model.nx) # BU (from Bt)
                

        # --- Time-domain regularization ---
        if model.nt == 1:
            # If there's only one time step, we don't apply temporal regularization
            total_temporal_loss = torch.tensor(0.0, device=parameters.device)
        else:
            total_temporal_loss = torch.tensor(0.0, device=parameters.device)
            # Slice parameters for temporal regularization: (n_pixels, n_time) for each param
            
            # Blos (index 0)
            total_temporal_loss += reguT_Blos * temporal_regularization(params_reshaped[:, :, 0])
            # BQ (index 1)
            total_temporal_loss += reguT_Bhor * temporal_regularization(params_reshaped[:, :, 1])
            # BU (index 2)
            total_temporal_loss += reguT_Bazi * temporal_regularization(params_reshaped[:, :, 2])


        # --- Total Loss ---
        loss = chi2loss + total_spatial_loss + total_temporal_loss

        loss.backward()
        
        # Normalize gradients:
        if normgrad:
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        p.grad = p.grad / (torch.mean(torch.abs(p.grad)) + 1e-9)
        
        optimizer.step()

        # print(total_temporal_loss.item())

        t.set_postfix({'total': loss.item(), 
                       'chi2': chi2loss.item(), 
                       'spatial': total_spatial_loss.item(),
                       'temporal': total_temporal_loss.item()})


    # Reshape the output parameters for plotting and return
    # outplot shape: (ny, nx, nTime, n_model_params)
    outplot = parameters.clone().detach().numpy().reshape(model.ny, model.nx, model.nt, parameters.shape[-1])
    
    # Calculate Bt and phiB for the output (these are derived from BQ and BU)
    # We need to do this for each time step
    Bt = np.sqrt(np.sqrt(outplot[:,:,:,2]**2. + outplot[:,:,:,1]**2)) # sqrt(sqrt(BU^2 + BQ^2))
    phiB = 0.5 * np.arctan2(outplot[:,:,:,2], outplot[:,:,:,1]) # atan2(BU, BQ)
    phiB[phiB < 0] += np.pi
    phiB[phiB > np.pi] -= np.pi # Ensure phiB is in [0, pi] range

    # Update outplot with Bt and phiB (original Blos is already correct)
    outplot_final = outplot.copy()
    outplot_final[:,:,:,1] = Bt
    outplot_final[:,:,:,2] = phiB

    return outplot_final, parameters 
