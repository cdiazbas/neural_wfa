import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys
from tqdm import tqdm, trange
import astropy.io.fits as fits
from einops import rearrange

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
    diffs = torch.diff(params_time_series, dim=-1)  # (n_pixels, n_time - 1)
    
    return torch.sum(diffs**2.0)
    # return torch.sum(torch.abs(diffs))


def prepare_initial_guess(model, use_temporal_average=False, bad_frames=None):
    """
    Prepares the initial guess tensor for model parameters.

    Args:
        model: The model object, expected to have attributes:
            - ny (int): Number of pixels in y-direction.
            - nx (int): Number of pixels in x-direction.
            - nt (int): Number of time steps.
            - initial_guess (callable): Function returning initial guesses for Blos, BQ, BU.
        use_temporal_average (bool): If True, compute the temporal average of the initial guess
                                     and replicate it across all time frames.
        bad_frames (list or array): List of frame indices to replace with temporal average.
                                   If provided, only these frames are replaced.

    Returns:
        torch.Tensor: Initial guess tensor of shape (ny * nx * nt, 3), with gradients enabled.
    """
    # Allocate tensor for parameters: (ny * nx * nt, 3)
    out = torch.ones((model.ny * model.nx * model.nt, 3), dtype=torch.float32)

    # Get initial guesses for Blos, BQ, BU (shape: (ny*nx*nt,))
    B0_exp, BtQ_exp, BtU_exp = model.initial_guess(inner=True, split=True)

    if use_temporal_average or bad_frames is not None:
        # Reshape to (nt, ny*nx)
        B0_reshaped = B0_exp.reshape(model.nt, model.ny * model.nx)
        BQ_reshaped = BtQ_exp.reshape(model.nt, model.ny * model.nx)
        BU_reshaped = BtU_exp.reshape(model.nt, model.ny * model.nx)
        
        # Compute temporal average: (ny*nx,)
        B0_avg = torch.mean(B0_reshaped, dim=0)
        BQ_avg = torch.mean(BQ_reshaped, dim=0)
        BU_avg = torch.mean(BU_reshaped, dim=0)
        
        if use_temporal_average:
            # Replicate across all time frames: (nt, ny*nx) -> (ny*nx*nt,)
            B0_exp = B0_avg.unsqueeze(0).repeat(model.nt, 1).flatten()
            BtQ_exp = BQ_avg.unsqueeze(0).repeat(model.nt, 1).flatten()
            BtU_exp = BU_avg.unsqueeze(0).repeat(model.nt, 1).flatten()
        elif bad_frames is not None:
            # Only replace bad frames with average
            for frame_idx in bad_frames:
                if 0 <= frame_idx < model.nt:
                    B0_reshaped[frame_idx, :] = B0_avg
                    BQ_reshaped[frame_idx, :] = BQ_avg
                    BU_reshaped[frame_idx, :] = BU_avg
            
            # Flatten back to (ny*nx*nt,)
            B0_exp = B0_reshaped.flatten()
            BtQ_exp = BQ_reshaped.flatten()
            BtU_exp = BU_reshaped.flatten()

    # Normalize and assign to tensor
    out[:, 0] = B0_exp / 1e3      # Blos
    out[:, 1] = BtQ_exp / 1e6     # BQ (from Bt and phiB)
    out[:, 2] = BtU_exp / 1e6     # BU (from Bt and phiB)

    # Enable gradients
    out.requires_grad = True

    return out

# =================================================================
def optimization(optimizer, niterations, parameters, model, 
                 reguV=0.0, reguQU=0.0, reguT_Blos=0.0, reguT_BQU=0.0, 
                 reguMag_Blos=0.0, reguMag_QU=0.0,
                 weights=[10,10,1], normgrad=False, mask=None, tweights=None):
    # Optimization loop
    
    # Explicit WFA
    model.Vnorm = 1000.0
    model.QUnorm = 1e6
    
    
    # Convert tweights to tensor if provided
    if tweights is not None:
        tweights_tensor = torch.as_tensor(tweights, dtype=torch.float32, device=parameters.device)
        if tweights_tensor.shape[0] != model.nt:
            raise ValueError(f"tweights length {tweights_tensor.shape[0]} must match nt={model.nt}")
        # Reshape to (1, nt) for broadcasting with (n_pixels, nt)
        tweights_tensor = tweights_tensor.view(1, -1)
    else:
        tweights_tensor = None

    
    # parameters shape: (ny * nx, nTime, n_model_params)

    t = trange(niterations, leave=True)
    for loop in t:
        optimizer.zero_grad()

        # Chi2 loss (fidelity to data)
        chi2loss = model.evaluate(parameters, weights=weights, spatial_mask=mask, tweights=tweights_tensor)

        # Reshape parameters for easier spatial/temporal regularization access
        # (ny * nx, nTime, n_model_params)

        # If there are multiple time steps, we reshape to (ny * nx, nt, n_model_params)
        # This allows us to access each time step's parameters easily

        # Using rearrange if needed:
        params_reshaped = rearrange(parameters, '(nt ny nx) p -> (ny nx) nt p', ny=model.ny, nx=model.nx, nt=model.nt)

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
            total_temporal_loss += reguT_BQU * temporal_regularization(params_reshaped[:, :, 1])
            # BU (index 2)
            total_temporal_loss += reguT_BQU * temporal_regularization(params_reshaped[:, :, 2])


        # --- Magnitude regularization (penalizes large parameter values) ---
        total_magnitude_loss = torch.tensor(0.0, device=parameters.device)
        if reguMag_Blos > 0:
            total_magnitude_loss += reguMag_Blos * torch.sum(params_reshaped[:, :, 0]**2)  # BV (Blos)
        if reguMag_QU > 0:
            total_magnitude_loss += reguMag_QU * torch.sum(params_reshaped[:, :, 1]**2)  # BQ
            total_magnitude_loss += reguMag_QU * torch.sum(params_reshaped[:, :, 2]**2)  # BU

        # --- Total Loss ---
        loss = chi2loss + total_spatial_loss + total_temporal_loss + total_magnitude_loss

        loss.backward()
        
        # Normalize gradients:
        if normgrad:
            for param_group in optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is not None:
                        p.grad = p.grad / (torch.mean(torch.abs(p.grad)) + 1e-9)
        
        optimizer.step()

        # Update the progress bar with the current loss values
        t.set_postfix({'total': loss.item(), 
                       'chi2': chi2loss.item(), 
                       'spatial': total_spatial_loss.item(),
                       'temporal': total_temporal_loss.item(),
                       'magnitude': total_magnitude_loss.item()})


    # Reshape the output parameters for plotting and return
    # outplot shape: (ny, nx, nTime, n_model_params)
    outplot = parameters.clone().detach().numpy()
    # Back to (nt, ny, nx, n_model_params)
    outplot = rearrange(outplot, '(nt ny nx) p -> nt ny nx p', ny=model.ny, nx=model.nx, nt=model.nt)
    
    # Revert the normalization in the output parameters:
    outplot[...,0] *= model.Vnorm  # Blos
    outplot[...,1] *= model.QUnorm  # BQ
    outplot[...,2] *= model.QUnorm  # BU

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
