import torch
import torch.optim as optim
import numpy as np
from tqdm import trange
import time

from neural_wfa.core.problem import WFAProblem
from neural_wfa.core.magnetic_field import MagneticField
from neural_wfa.regularization.spatial import smoothness_loss
from neural_wfa.regularization.temporal import temporal_smoothness_loss

class PixelSolver:
    """
    Solver for Explicit (Pixel-wise) WFA Inversion.
    
    Optimizes magnetic field parameters directly for each pixel (and time step).
    """
    def __init__(
        self,
        problem: WFAProblem,
        nt: int = None,
        device: torch.device = None
    ):
        self.problem = problem
        self.nt = nt if nt is not None else problem.nt
        self.device = device if device else problem.device
        
        # Dimensions from observation
        if len(problem.obs.grid_shape) == 2:
            self.ny, self.nx = problem.obs.grid_shape
        else:
            self.ny = problem.obs.n_pixels
            self.nx = 1 # Treat as flat list if not grid
        
        self.n_pixels = self.ny * self.nx
        
        # Parameters: (N_pixels, N_time, 3)
        # 0: Blos (normalized)
        # 1: BQ (normalized)
        # 2: BU (normalized)
        # We start with zeros or heuristics
        self.params = torch.zeros(self.n_pixels, self.nt, 3, device=self.device, requires_grad=True)
        
        # Normalization constants
        self.Vnorm = 1000.0
        self.QUnorm = 1000.0 # Legacy used 1000 or 1e6?
        # explicit.py: model.QUnorm = 1e6. 
        # But bfield.py WFA_model3D sets QUnorm = 1000 in init?
        # Let's check Baseline Explicit code.
        # It calls `model.QUnorm = 1e6` inside `optimization` function? NO.
        # `models/explicit.py` line 102: `model.QUnorm = 1e6`.
        # So explicit uses 1e6 for Q/U normalization.
        # But `WFA_model3D` uses `self.QUnorm = 1000.0` by default.
        # Why the difference?
        # Explicit solver operates on "normalized parameters" that are small (~1).
        # BQ ~ 1000 G. If Norm=1e6, param ~ 0.001.
        # If Norm=1000, param ~ 1.
        # Explicit solver initial guess: B / 1e3 (Blos), B / 1e6 (BQ).
        # So Params are expected to be O(1) or O(0.001).
        # I'll stick to 1e6 for BQ/BU if replicating Explicit.
        
        self.Vnorm = 1000.0
        self.QUnorm = 1e6
        
        # Kernel for Legacy Spatial Regularization
        self._spatial_kernel = torch.tensor([[0.5, 1.0, 0.5],
                                            [1.0, 0.0, 1.0],
                                            [0.5, 1.0, 0.5]], device=self.device)
        self._spatial_kernel = self._spatial_kernel / self._spatial_kernel.sum()
        self._spatial_kernel = self._spatial_kernel.view(1, 1, 3, 3)

    def _spatial_regularization(self, param_2d):
        """
        Legacy spatial regularization: Squared diff from weighted neighbor average.
        param_2d: (1, 1, Ny, Nx) or (Batch, 1, Ny, Nx)
        """
        # Padding
        # ReflectionPad2d(1) handles edges like Legacy
        from torch.nn.functional import conv2d, pad
        
        # Legacy uses nn.ReflectionPad2d(1) -> conv2d(..., padding='valid')
        # Here we pad manually or use pad
        padded = pad(param_2d, (1, 1, 1, 1), mode='reflect')
        output_smooth = conv2d(padded, self._spatial_kernel, padding=0)
        
        # Squared difference
        return torch.sum(torch.abs(output_smooth - param_2d)**2.0)

    def _temporal_regularization(self, params_time_series):
        """
        Legacy temporal regularization: Sum of squared diffs.
        params_time_series: (N_pixels, N_time)
        """
        diffs = torch.diff(params_time_series, dim=-1)
        return torch.sum(torch.abs(diffs)**2.0)

    def initialize_parameters(self, method='weak_field'):
        if method == 'zeros':
            with torch.no_grad():
                self.params.fill_(0.0)
                self.params.add_(torch.randn_like(self.params) * 1e-5)
        elif method == 'weak_field':
            with torch.no_grad():
                Blos, BQ, BU = self.problem.compute_initial_guess()
                
                # Normalize
                blos_norm = Blos / self.Vnorm
                bq_norm = BQ / self.QUnorm
                bu_norm = BU / self.QUnorm
                
                # Handle Time Dimension if params has T
                # compute_initial_guess returns (N,) or (N, T) depending on input.
                # If problem.obs is (N,L) (T=1 implicit), output is (N).
                # self.params is (N, T, 3).
                
                if blos_norm.ndim == 1 and self.nt == 1:
                     self.params[:, 0, 0] = blos_norm
                     self.params[:, 0, 1] = bq_norm
                     self.params[:, 0, 2] = bu_norm
                elif blos_norm.ndim == 1 and self.nt > 1:
                     # Repeat over time? Or assume it's static guess?
                     for t in range(self.nt):
                         self.params[:, t, 0] = blos_norm
                         self.params[:, t, 1] = bq_norm
                         self.params[:, t, 2] = bu_norm
                else:
                     # Assume dimensions match
                     self.params[:, :, 0] = blos_norm
                     self.params[:, :, 1] = bq_norm
                     self.params[:, :, 2] = bu_norm
        
    def solve(
        self, 
        n_iterations: int = 100, 
        lr: float = 1e-2,
        reguV: float = 1e-3,
        reguQU: float = 0.5e-1,
        reguT_Blos: float = 1e-3,
        reguT_Bhor: float = 1e-3,
        reguT_Bazi: float = 1e-3,
        verbose: bool = True
    ):
        """
        Runs the optimization loop.
        """
        optimizer = optim.Adam([self.params], lr=lr)
        # Legacy used Adam or LBFGS? explicit.py uses Adam.
        
        t = trange(n_iterations, leave=True)

        for i in t:
            optimizer.zero_grad()
            
            t0 = time.time()
            
            # --- DIRECT CALCULATION (Optimized for speed) ---
            # Unpack parameters
            blos_norm = self.params[:, :, 0] # (N, T)
            bq_norm   = self.params[:, :, 1]
            bu_norm   = self.params[:, :, 2]
            
            Blos = blos_norm * self.Vnorm # (N, T)
            BQ = bq_norm * self.QUnorm
            BU = bu_norm * self.QUnorm
            
            # Constants from Problem
            C = self.problem.C
            geff = self.problem.lin.geff
            Gg = self.problem.lin.Gg
            
            # Derivatives (assume full batch)
            dIdw = self.problem.dIdw # (N, L)
            dIdwscl = self.problem.dIdwscl
            
            # Helper for broadcasting over Time dimension
            if self.nt > 1:
                # Blos: (N, T) -> (N, T, 1)
                Blos_b = Blos.unsqueeze(-1)
                BQ_b = BQ.unsqueeze(-1)
                BU_b = BU.unsqueeze(-1)
                
                # Derivatives: (N, L) -> (N, 1, L)
                dIdw_b = dIdw.unsqueeze(1)
                dIdwscl_b = dIdwscl.unsqueeze(1)
            else:
                # Blos: (N, 1). Derivatives: (N, L). Broadcasting works directly (N, L)
                Blos_b = Blos
                BQ_b = BQ
                BU_b = BU
                dIdw_b = dIdw
                dIdwscl_b = dIdwscl
                
            # Forward Model
            stokesV = C * geff * Blos_b * dIdw_b
            Clp = 0.75 * (C**2) * Gg * dIdwscl_b
            stokesQ = Clp * BQ_b
            stokesU = Clp * BU_b
            
            # Loss Calculation
            # Obs: (N_pixels, N_lambda) or (N_pixels, N_time, N_lambda)?
            # Observation class usually standardizes to something compatible?
            # If Observation has T dim, calculate against it.
            # Here assuming Obs matches Model output or is broadcastable.
            
            obs_Q = self.problem.obs.stokes_Q
            obs_U = self.problem.obs.stokes_U
            obs_V = self.problem.obs.stokes_V
            
            if self.nt > 1 and obs_Q.ndim == 2:
                 # Obs is (N, L), Model is (N, T, L). 
                 # We probably want to compare Model vs Obs?
                 # If Obs is single-frame but we model T frames, we assume Obs is const or T=1?
                 # If Obs is movie, Observation should store it as (N, T, L).
                 # Current Observation implementation stores (N_pixels, N_lambda).
                 # If T>1, N_pixels includes T dimension?
                 # PixelSolver init: self.ny, self.nx = problem.obs.shape_spatial
                 # self.n_pixels = ny * nx.
                 # If Obs is flattened (N*T), then self.nt should represent T?
                 # Legacy handles T implicitly via reshaping. 
                 # Let's assume Obs aligns with (N, L) for now as typical WFA is per-pixel.
                 # If T>1, maybe Obs should broadcast?
                 obs_Q = obs_Q.unsqueeze(1)
                 obs_U = obs_U.unsqueeze(1)
                 obs_V = obs_V.unsqueeze(1)
            
            mask = self.problem.active_wav_idx # Spectral indices
            
            # --- LEGACY PARITY: Use L1 Loss (Mean Absolute Error) ---
            # Legacy evaluate: weights[0]*mean(|dQ|) + weights[1]*mean(|dU|) + weights[2]*mean(|dV|)
            # Where mean is over pixels (and wavelengths implicitly via flattening or manual mean)
            # Legacy code takes 2D mean? 
            # `torch.mean(torch.abs(self.data_stokesQ - stokesQ)[:, self.mask])`
            # Wait, `self.mask` in legacy `evaluate` is spatial mask?
            # "mask: [5, 6, 7] are the indices to use during the optimization" (from run output).
            # This is SPATIAL indices.
            # My `self.problem.mask` is SPECTRAL indices in my implementation?
            # Let's check `WFAProblem`. `self.active_wav_idx = mask`
            # In `profile_solver.py`: `obs = Observation(..., active_wav_idx=[5,6,7])`. 
            # In `run_example_explicit.py`: `mask = [5, 6, 7]`. passed to `WFA_model3D`.
            # `WFA_model3D` uses `self.mask` to slice SPATIALLY.
            # `evaluate` does `mean(abs(diff)[:, self.mask])` -> Selects columns?
            # Data shape (Ny*Nx, Nv). self.mask selects wavelengths?
            # WFA_model3D init: `if mask is None: mask = range(self.data.shape[-1])`. 
            # data shape is (Ny*Nx, Ns, Nw). 
            # But in `forward` indices is `range(0, len(self.dIdw))`. 
            # Actually, `self.mask` in `WFA_model3D` IS SPECTRAL (wavelengths) mask.
            # "mask: [5, 6, 7] are the indices to use during the optimization".
            # OK, so my `self.problem.active_wav_idx` is consistent (Spectral).
            
            # Legacy: Mean absolute difference over selected wavelengths.
            diff_V = torch.abs(obs_V - stokesV)[..., mask]
            diff_Q = torch.abs(obs_Q - stokesQ)[..., mask]
            diff_U = torch.abs(obs_U - stokesU)[..., mask]
            
            # Mean over everything (pixels + wavelengths)
            mae_V = torch.mean(diff_V)
            mae_Q = torch.mean(diff_Q)
            mae_U = torch.mean(diff_U)
            
            # Weights: Legacy defaults [10, 10, 1] in optimization function signature.
            # WFAProblem stores weights.
            w = self.problem.weights # [10, 10, 1] usually
            
            chi2_loss = w[0] * mae_Q + w[1] * mae_U + w[2] * mae_V
            t1 = time.time()
            
            # 3. Regularization
            reg_loss = torch.tensor(0.0, device=self.device)
            # ... (Rest of regularization skipped for brevity in print, assuming it works)
            
            # Spatial - need reshaping to (Ny, Nx)
            # Spatial - need reshaping to (Batch, 1, Ny, Nx) for conv2d
            # PixelSolver stores params as (N_pixels, N_t, 3).
            # We treat each Time slice independently for Spatial reg.
            
            if reguV > 0 or reguQU > 0:
                 # Reshape to (Ny, Nx, Nt, 3)
                 params_grid = self.params.view(self.ny, self.nx, self.nt, 3)
                 
                 for t_idx in range(self.nt):
                     # Extract (Ny, Nx, 3)
                     p_t = params_grid[:, :, t_idx, :]
                     
                     if reguV > 0:
                         # (1, 1, Ny, Nx)
                         blos_map = p_t[:, :, 0].unsqueeze(0).unsqueeze(0)
                         reg_loss += reguV * self._spatial_regularization(blos_map)
                         
                     if reguQU > 0:
                         bq_map = p_t[:, :, 1].unsqueeze(0).unsqueeze(0)
                         bu_map = p_t[:, :, 2].unsqueeze(0).unsqueeze(0)
                         reg_loss += reguQU * self._spatial_regularization(bq_map)
                         # Note: Legacy adds reguQU * regularize(BU) separately. 
                         # And usesSAME coefficient reguQU for both.
                         reg_loss += reguQU * self._spatial_regularization(bu_map)

            # Temporal
            if self.nt > 1:
                if reguT_Blos > 0: reg_loss += reguT_Blos * self._temporal_regularization(self.params[..., 0])
                if reguT_Bhor > 0: reg_loss += reguT_Bhor * self._temporal_regularization(self.params[..., 1])
                if reguT_Bazi > 0: reg_loss += reguT_Bazi * self._temporal_regularization(self.params[..., 2])

            total_loss = chi2_loss + reg_loss
            
            total_loss.backward()
            t2 = time.time()
            
            if self.params.grad is not None:
                 self.params.grad /= (torch.mean(torch.abs(self.params.grad)) + 1e-9)
            
            optimizer.step()
            t3 = time.time()
            
            t.set_postfix({'total': total_loss.item(), 
                           'chi2': chi2_loss.item(), 
                           'spatial': reg_loss.item(),
                           'temporal': 0.0}) # if nt=1
                
    def get_field(self) -> MagneticField:
        with torch.no_grad():
           blos_norm = self.params[:, :, 0]
           bqu_packed = torch.stack([self.params[:, :, 1], self.params[:, :, 2]], dim=-1)
           return MagneticField(blos_norm, bqu_packed, w_blos=self.Vnorm, w_bqu=self.QUnorm, grid_shape=self.problem.obs.grid_shape)
