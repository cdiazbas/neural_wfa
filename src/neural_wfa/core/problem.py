import torch
import numpy as np
from neural_wfa.core.magnetic_field import MagneticField
from neural_wfa.core.observation import Observation
from neural_wfa.physics.lines import LineInfo
from neural_wfa.physics.derivatives import cder


class WFAProblem:
    """
    Defines the Weak Field Approximation (WFA) optimization problem.

    This class handles the forward physics model (Magnetic Field ->
    Stokes Vectors) and the loss computation (Model vs Observation).
    """

    def __init__(
        self,
        observation: Observation,
        line_info: LineInfo,
        active_wav_idx: torch.Tensor = None,
        weights: list = [1.0, 1.0, 1.0],  # Weights for [Q, U, V]
        device: torch.device = None,
        verbose: bool = True,
        vdop: float = 0.035,
    ):
        self.obs = observation
        self.lin = line_info
        self.device = device if device else observation.device
        self.nt = getattr(observation, "nt", 1)
        self.vdop = vdop

        # Move observation to device if not already
        if self.obs.device != self.device:
            self.obs = self.obs.to(self.device)

        # Weights for loss function
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Active spectral indices (from Obs or overridden)
        self.active_wav_idx = (
            active_wav_idx.to(self.device)
            if active_wav_idx is not None
            else self.obs.active_wav_idx
        )

        if verbose:
            print(
                "Data:",
                self.obs.flat_data.shape,
                "should be in the format [(nt) ny nx ns nw] (flattened to [N ns nw])",
            )
            print(
                "Wav:",
                self.obs.wavelengths.shape,
                "should be in Angstroms relative to the center of the line",
            )
            print(
                "mask:",
                self.active_wav_idx,
                "are the indices to use during the optimization",
            )

        # Precompute derivatives dI/dlambda
        self._compute_derivatives()

        # Physical constants
        # C = -4.67e-13 * lambda0^2 (in Angstrom)
        self.C = -4.67e-13 * (self.lin.cw**2)

    def compute_initial_guess(self, indices=None):
        """
        Computes the algebraic Weak Field Approximation solution. This provides
        a very good initialization for the optimizer.

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
            active_idx = self.active_wav_idx
        else:
            stokesQ = self.obs.stokes_Q[indices]
            stokesU = self.obs.stokes_U[indices]
            stokesV = self.obs.stokes_V[indices]
            dIdw = self.dIdw[indices]
            dIdwscl = self.dIdwscl[indices]
            active_idx = self.active_wav_idx

        # dIdw is usually (N, L). If obs is (N, T, L), we broadcast dIdw.
        if stokesV.ndim == 3 and dIdw.ndim == 2:
            dIdw = dIdw.unsqueeze(1)
            dIdwscl = dIdwscl.unsqueeze(1)

        # Apply active indices
        idx_mask = active_idx if active_idx is not None else slice(None)

        # Blos = Sum(V * dI) / (C * geff * Sum(dI^2))
        sliced_V = stokesV[..., idx_mask]
        sliced_dIdw = dIdw[..., idx_mask]

        numer_blos = torch.sum(sliced_V * sliced_dIdw, dim=-1)
        denom_blos = self.C * self.lin.geff * torch.sum(sliced_dIdw**2, dim=-1)
        Blos = numer_blos / denom_blos  # No epsilon to match legacy

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

        stokes_I.detach().cpu().numpy()
        wavs_np = wavs.detach().cpu().numpy()

        # Obs.flat_data is (N_pixels, 4, N_lambda)
        full_data_np = (
            self.obs.flat_data.detach().cpu().numpy()[None, ...]
        )  # Add dummy dim for N1

        dIdw_np = cder(wavs_np, full_data_np)
        # Result shape (1, N_pixels, N_lambda). Squeeze first dim.
        dIdw_np = dIdw_np[0]

        self.dIdw = torch.from_numpy(dIdw_np).to(self.device)

        # Second derivative scaling term for transverse (Q/U)
        # Legacy: dIdwscl = dIdw * scl
        # scl = 1.0 / (wl + 1e-9). Zeroed where |wl| <= vdop

        scl = 1.0 / (wavs_np + 1e-9)
        scl[np.abs(wavs_np) <= self.vdop] = 0.0
        scl_tensor = (
            torch.from_numpy(scl.astype(np.float32)).to(self.device).reshape(1, -1)
        )  # Broadcast over pixels if needed?
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
        active_blos: bool = True,
        active_bqu: bool = True,
        indices: torch.Tensor = None,
        spatial_weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Computes the weighted Chi-squared loss.

        Args:
            field (MagneticField): Current field estimate (Batch or Full).
            active_blos (bool): If False, zeros out Blos contribution.
            active_bqu (bool): If False, zeros out BQ/U contribution.
            indices (torch.Tensor): Indices of pixels in the batch.
            spatial_weights (torch.Tensor): Pixel-wise weights (shape (N_batch,) or (N_full,)).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        stokesQ_model, stokesU_model, stokesV_model = self.compute_forward_model(
            field, indices=indices
        )

        # Get Observation Data (Sliced if indices provided)
        if indices is not None:
            # Flattened access
            # Obs.stokes_Q is (N_pixels, N_lambdas)
            obs_Q = self.obs.stokes_Q[indices]
            obs_U = self.obs.stokes_U[indices]
            obs_V = self.obs.stokes_V[indices]

            # Mask is Spectral, so do NOT slice with spatial indices.
            # mask_spat = self.mask[indices] if self.mask is not None else None
        else:
            obs_Q = self.obs.stokes_Q
            obs_U = self.obs.stokes_U
            obs_V = self.obs.stokes_V
            # mask_spat = self.mask

        # Use active_wav_idx
        spectral_indices = (
            self.active_wav_idx if self.active_wav_idx is not None else slice(None)
        )

        res_V = ((obs_V - stokesV_model) ** 2)[..., spectral_indices]
        res_Q = ((obs_Q - stokesQ_model) ** 2)[..., spectral_indices]
        res_U = ((obs_U - stokesU_model) ** 2)[..., spectral_indices]

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
        if active_blos:
            loss += self.weights[2] * torch.mean(mse_V)
        if active_bqu:
            loss += self.weights[0] * torch.mean(mse_Q) + self.weights[1] * torch.mean(
                mse_U
            )

        return loss
