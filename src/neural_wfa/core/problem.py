import torch
import numpy as np
from neural_wfa.core.magnetic_field import MagneticField
from neural_wfa.core.observation import Observation
from neural_wfa.physics.lines import LineInfo
from neural_wfa.physics.derivatives import cder

class WFAProblem:
    """
    Defines the Weak Field Approximation (WFA) optimization problem.
    
    This class handles the forward physics model (Magnetic Field -> Stokes Vectors)
    and the loss computation (Model vs Observation).
    """
    def __init__(
        self, 
        observation: Observation, 
        line_info: LineInfo, 
        mask: torch.Tensor = None,
        weights: list = [1.0, 1.0, 1.0], # Weights for [Q, U, V]
        device: torch.device = None
    ):
        self.obs = observation
        self.lin = line_info
        self.device = device if device else observation.device
        self.nt = getattr(observation, 'nt', 1)
        
        # Move observation to device if not already
        if self.obs.device != self.device:
            self.obs = self.obs.to(self.device)
            
        # Weights for loss function
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Spatial mask (optional) - Shape (N_pixels,) or broadcastable
        self.mask = mask.to(self.device) if mask is not None else None
        
        # Precompute derivatives dI/dlambda
        self._compute_derivatives()
        
        # Physical constants
        # C = -4.67e-13 * lambda0^2 (in Angstrom)
        # Note: LineInfo.larm is 4.668...e-13.
        # Legacy code uses -4.67e-13 explicitly in WFA_model3D. 
        # Let's use the legacy constant to ensure exact numerical match for now, 
        # or stick to LineInfo.larm if we want consistency.
        # Legacy: self.C = -4.67e-13 * self.lin.cw**2
        self.C = -4.67e-13 * (self.lin.cw**2)
        
    def compute_initial_guess(self, indices=None):
        """
        Computes the algebraic Weak Field Approximation solution.
        This provides a very good initialization for the optimizer.
        
        Returns:
            tuple: (Blos, BQ, BU) physical values.
        """
        if indices is None:
            # Use full dataset
            stokesQ = self.obs.stokes_Q
            stokesU = self.obs.stokes_U
            stokesV = self.obs.stokes_V
            dIdw = self.dIdw
            dIdwscl = self.dIdwscl
            mask = self.mask
        else:
            stokesQ = self.obs.stokes_Q[indices]
            stokesU = self.obs.stokes_U[indices]
            stokesV = self.obs.stokes_V[indices]
            dIdw = self.dIdw[indices]
            dIdwscl = self.dIdwscl[indices]
            mask = self.mask

        # Broadcast if needed (N, N_lambda) vs (N_pixels, N_time, N_lambda) logic
        # For initial guess, we usually assume single time or treat Time as batch.
        # But stokes* might be (N, T, L) or (N, L).
        # We perform sum over Lambda.
        
        # dIdw is usually (N, L). If obs is (N, T, L), we broadcast dIdw.
        if stokesV.ndim == 3 and dIdw.ndim == 2:
             dIdw = dIdw.unsqueeze(1)
             dIdwscl = dIdwscl.unsqueeze(1)

        # Apply mask
        idx_mask = mask if mask is not None else slice(None)
        
        # Helper to sum over last dim (wavelength)
        # Helper to sum over last dim (wavelength) with explicit slicing
        # wsum(x) replaced by direct op for clarity/parity
        
        # Blos = Sum(V * dI) / (C * geff * Sum(dI^2))
        sliced_V = stokesV[..., idx_mask]
        sliced_dIdw = dIdw[..., idx_mask]
        
        numer_blos = torch.sum(sliced_V * sliced_dIdw, dim=-1)
        denom_blos = self.C * self.lin.geff * torch.sum(sliced_dIdw**2, dim=-1)
        Blos = numer_blos / denom_blos # No epsilon to match legacy

        # Constant for BQ/BU
        # C2G = 0.75 * C^2 * Gg
        C2G = 0.75 * (self.C**2) * self.lin.Gg
        
        sliced_dIdwscl = dIdwscl[..., idx_mask]
        sliced_Q = stokesQ[..., idx_mask]
        sliced_U = stokesU[..., idx_mask]
        
        denom_bqu = C2G * torch.sum(sliced_dIdwscl**2, dim=-1)
        
        # BQ = Sum(Q * dIscl) / Denom
        numer_bq = torch.sum(sliced_Q * sliced_dIdwscl, dim=-1)
        BQ = numer_bq / denom_bqu
        
        # BU = Sum(U * dIscl) / Denom
        numer_bu = torch.sum(sliced_U * sliced_dIdwscl, dim=-1)
        BU = numer_bu / denom_bqu
        
        return Blos, BQ, BU
        
    def _compute_derivatives(self):
        """
        Computes dI/dlambda needed for WFA.
        Stores them as tensors on the correct device.
        """
        # Get flattened Stokes I and wavelength
        # Shape (N_pixels, N_lambda)
        stokes_I = self.obs.stokes_I
        wavs = self.obs.wavelengths
        
        # cder expects numpy arrays on CPU usually? Let's check cder implementation.
        # My cder implementation uses numpy operations.
        # So we need to move to cpu/numpy, calculate, then move back.
        
        stokes_I_np = stokes_I.detach().cpu().numpy()
        wavs_np = wavs.detach().cpu().numpy()
        
        # cder supports 4D input or requires adaptation.
        # My cder implementation:
        # N1, N2, nstokes, nlam = y.shape
        # Input y should be (1, N_pixels, 1, N_lambda) to differentiate just I?
        # Or I can pass full stokes (1, N_pixels, 4, N_lambda).
        # Let's pass full stokes for compatibility with current cder signature.
        
        # Reshape flat data to (1, N_pixels, 4, N_lambda)
        # Obs.flat_data is (N_pixels, 4, N_lambda)
        full_data_np = self.obs.flat_data.detach().cpu().numpy()[None, ...] # Add dummy dim for N1
        
        dIdw_np = cder(wavs_np, full_data_np)
        # Result shape (1, N_pixels, N_lambda). Squeeze first dim.
        dIdw_np = dIdw_np[0] 
        
        self.dIdw = torch.from_numpy(dIdw_np).to(self.device)
        
        # Second derivative scaling term for transverse (Q/U)
        # Legacy: dIdwscl = dIdw * scl
        # scl = 1.0 / (wl + 1e-9). Zeroed where |wl| <= vdop (0.035)
        # Legacy uses vdop=0.035 hardcoded in __init__? Yes.
        vdop = 0.035
        
        scl = 1.0 / (wavs_np + 1e-9)
        scl[np.abs(wavs_np) <= vdop] = 0.0
        scl_tensor = torch.from_numpy(scl.astype(np.float32)).to(self.device).reshape(1, -1) # Broadcast over pixels if needed? 
        # wavs is 1D, so scl is 1D.
        
        self.dIdwscl = self.dIdw * scl_tensor

    def compute_forward_model(self, field: MagneticField, indices: torch.Tensor = None):
        """
        Computes modeled Stokes Q, U, V given the magnetic field.
        
        Args:
            field (MagneticField): The magnetic field state.
            indices (torch.Tensor): Indices of pixels corresponding to the field batch.
            
        Returns:
            Tuple[Tensor, Tensor, Tensor]: (stokesQ, stokesU, stokesV)
            Shapes are (N_pixels, N_time, N_lambda) or (Batch, N_lambda).
        """
        
        # Get physical quantities
        Blos = field.blos
        BQ_val = field.b_q 
        BU_val = field.b_u 
        
        # Prepare derivatives and constants (Sliced if needed)
        # dIdw: (N_pixels, N_lambda)
        # dIdwscl: (N_pixels, N_lambda)
        if indices is not None:
            dIdw = self.dIdw[indices]
            dIdwscl = self.dIdwscl[indices]
        else:
            dIdw = self.dIdw
            dIdwscl = self.dIdwscl
            
        # Handle broadcasting (Time dimension or Batch)
        # Blos shape could be (N,) or (N, T) or (Batch,)
        # dIdw shape is (N, L)
        
        # If Blos has extra dimension (Time), it is (N, T).
        # dIdw needs to be (N, 1, L).
        # We assume 0-th dimension always corresponds to spatial index (pixel or batch index).
        
        if Blos.ndim > 1:
            # Assume shape (N, T) -> Unsqueeze dIdw to (N, 1, L)
            dIdw = dIdw.unsqueeze(1)
            dIdwscl = dIdwscl.unsqueeze(1)
            
            # Blos -> (N, T, 1)
            Blos_b = Blos.unsqueeze(-1)
            BQ_b = BQ_val.unsqueeze(-1)
            BU_b = BU_val.unsqueeze(-1)
        else:
            # Shape (N,) -> (N, 1)
            Blos_b = Blos.unsqueeze(-1)
            BQ_b = BQ_val.unsqueeze(-1)
            BU_b = BU_val.unsqueeze(-1)
            
        stokesV = self.C * self.lin.geff * Blos_b * dIdw
        
        Clp = 0.75 * (self.C**2) * self.lin.Gg * dIdwscl
        stokesQ = Clp * BQ_b
        stokesU = Clp * BU_b
        
        return stokesQ, stokesU, stokesV

    def compute_loss(
        self, 
        field: MagneticField, 
        mask_blos: bool = True, 
        mask_bqu: bool = True,
        indices: torch.Tensor = None,
        spatial_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Computes the weighted Chi-squared loss.
        
        Args:
            field (MagneticField): Current field estimate (Batch or Full).
            mask_blos (bool): If False, zeros out Blos contribution.
            mask_bqu (bool): If False, zeros out BQ/U contribution.
            indices (torch.Tensor): Indices of pixels in the batch. 
            spatial_weights (torch.Tensor): Pixel-wise weights (shape (N_batch,) or (N_full,)).
            
        Returns:
            torch.Tensor: Scalar loss value.
        """
        stokesQ_model, stokesU_model, stokesV_model = self.compute_forward_model(field, indices=indices)
        
        # Get Observation Data (Sliced if indices provided)
        if indices is not None:
            # Flattened access
            # Obs.stokes_Q is (N_pixels, N_lambdas)
            obs_Q = self.obs.stokes_Q[indices]
            obs_U = self.obs.stokes_U[indices]
            obs_V = self.obs.stokes_V[indices]
            
            # Mask is Spectral, so do NOT slice with spatial indices.
            # mask_spat = self.mask[indices] if self.mask is not None else None
            pass
        else:
            obs_Q = self.obs.stokes_Q
            obs_U = self.obs.stokes_U
            obs_V = self.obs.stokes_V
            # mask_spat = self.mask
        
        # Residuals
        # diffQ = torch.abs(obs_Q - stokesQ_model)
        # diffU = torch.abs(obs_U - stokesU_model)
        # diffV = torch.abs(obs_V - stokesV_model)
        
        # Apply spatial mask
        # if mask_spat is not None:
        #     # mask shape (N_pixels, 1) or broadcastable
        #     diffQ = diffQ * mask_spat
        #     diffU = diffU * mask_spat
        #     diffV = diffV * mask_spat
            
        # Mean over pixels and wavelengths
        # lossQ = torch.mean(diffQ) * self.weights[0]
        # lossU = torch.mean(diffU) * self.weights[1]
        # lossV = torch.mean(diffV) * self.weights[2]
        
        # total_loss = 0.0
        # if mask_bqu:
        #     total_loss += lossQ + lossU
        # if mask_blos:
        #     total_loss += lossV
            
        # return total_loss

        # Apply spectral mask if Obs has it (Obs.mask is wavelength indices)
        # Obs.Q is already full shape. 
        # If Obs has self.mask_indices, we should only compute loss on those indices.
        # For now, simplistic approach: use all wavelengths provided in obs tensors.
        
        # Note: The original self.mask is a spatial mask.
        # The following lines assume self.mask is a spectral mask (wavelength indices).
        # If self.mask is None, this will cause an error.
        # If self.mask is a spatial mask, this will cause an error.
        # Assuming self.mask is intended to be a spectral mask for this section.
        spectral_mask_for_loss = self.mask if self.mask is not None else slice(None)

        res_V = ((obs_V - stokesV_model)**2)[..., spectral_mask_for_loss]
        res_Q = ((obs_Q - stokesQ_model)**2)[..., spectral_mask_for_loss]
        res_U = ((obs_U - stokesU_model)**2)[..., spectral_mask_for_loss]
        
        # Reduced per pixel (mean over wavelengths)
        mse_V = torch.mean(res_V, dim=-1)
        mse_Q = torch.mean(res_Q, dim=-1)
        mse_U = torch.mean(res_U, dim=-1)
        
        # Apply spatial weights if any
        if spatial_weights is not None:
            mse_V = mse_V * spatial_weights
            mse_Q = mse_Q * spatial_weights
            mse_U = mse_U * spatial_weights
            
        # Final weighted sum
        loss = 0.0
        if mask_blos:
            loss += self.weights[2] * torch.mean(mse_V)
        if mask_bqu:
            loss += self.weights[0] * torch.mean(mse_Q) + self.weights[1] * torch.mean(mse_U)
            
        return loss
