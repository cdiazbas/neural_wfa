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
    ):
        self.problem = problem
        self.nt = problem.nt
        self.device = device if device else problem.device
        
        # Determine spatial dimensions vs total dimensions
        obs_shape = problem.obs.grid_shape
        if len(obs_shape) == 3: # (Nt, Ny, Nx)
            self.ny = obs_shape[1]
            self.nx = obs_shape[2]
            self.n_spatial = self.ny * self.nx
        elif len(obs_shape) == 2: # (Ny, Nx) -> nt=1
            self.ny = obs_shape[0]
            self.nx = obs_shape[1]
            self.n_spatial = self.ny * self.nx
        else:
            # Fallback for flat lists or unstructured
            self.n_spatial = problem.obs.n_pixels // self.nt
            self.ny = self.n_spatial
            self.nx = 1
        
        self.n_pixels = self.problem.obs.n_pixels # Total pixels (Nt * Ns)
        
        # Parameters: (N_time, N_spatial, 3)
        # Ordering matches Observation flattening (Time is slowest dim in C-order reshape of (Nt, Ny, Nx))
        # 0: Blos (normalized)
        # 1: BQ (normalized)
        # 2: BU (normalized)
        self.params = torch.zeros(self.nt, self.n_spatial, 3, device=self.device, requires_grad=True)
        
        # Normalization constants
        self.Vnorm = 1000.0
        self.QUnorm = 1e6
        

    def initialize_parameters(self, method='weak_field'):
        if method == 'zeros':
            with torch.no_grad():
                self.params.fill_(0.0)
                self.params.add_(torch.randn_like(self.params) * 1e-5)
        elif method == 'weak_field':
            with torch.no_grad():
                # Returns flat arrays (N_total,)
                Blos, BQ, BU = self.problem.compute_initial_guess()
                
                # Check consistency
                if Blos.numel() != self.nt * self.n_spatial:
                    print(f"Warning: Initial guess shape {Blos.shape} does not match parameter shape ({self.nt}, {self.n_spatial}). Using flat fill.")
                
                # Reshape to (Nt, Ns) and normalize
                self.params[:, :, 0] = Blos.view(self.nt, self.n_spatial) / self.Vnorm
                self.params[:, :, 1] = BQ.view(self.nt, self.n_spatial) / self.QUnorm
                self.params[:, :, 2] = BU.view(self.nt, self.n_spatial) / self.QUnorm
        
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
        
        t = trange(n_iterations, leave=True)

        for i in t:
            optimizer.zero_grad()
            
            # --- DIRECT CALCULATION (Optimized for speed) ---
            # Flatten parameters to (N_total, 3) to match Physics/Observation
            # params is (Nt, Ns, 3) -> reshape(-1, 3) -> (Nt*Ns, 3)
            
            p_flat = self.params.reshape(-1, 3)
            
            Blos = p_flat[:, 0] * self.Vnorm
            BQ = p_flat[:, 1] * self.QUnorm
            BU = p_flat[:, 2] * self.QUnorm
            
            # Constants from Problem
            C = self.problem.C
            geff = self.problem.lin.geff
            Gg = self.problem.lin.Gg
            
            # Derivatives (N_total, L)
            dIdw = self.problem.dIdw
            dIdwscl = self.problem.dIdwscl
            
            # Forward Model (Elementwise on Flattened Arrays)
            # Blos: (N_total). dIdw: (N_total, L). unsqueeze needed for broadcasting
            stokesV = C * geff * Blos.unsqueeze(-1) * dIdw
            
            Clp = 0.75 * (C**2) * Gg * dIdwscl
            stokesQ = Clp * BQ.unsqueeze(-1)
            stokesU = Clp * BU.unsqueeze(-1)
            
            # Loss Calculation (Observation is already flat N_total)
            obs_Q = self.problem.obs.stokes_Q
            obs_U = self.problem.obs.stokes_U
            obs_V = self.problem.obs.stokes_V
            
            mask = self.problem.active_wav_idx
            
            diff_V = torch.abs(obs_V - stokesV)[..., mask]
            diff_Q = torch.abs(obs_Q - stokesQ)[..., mask]
            diff_U = torch.abs(obs_U - stokesU)[..., mask]
            
            mae_V = torch.mean(diff_V)
            mae_Q = torch.mean(diff_Q)
            mae_U = torch.mean(diff_U)
            
            w = self.problem.weights
            chi2_loss = w[0] * mae_Q + w[1] * mae_U + w[2] * mae_V
            
            # 3. Regularization
            loss_spatial = torch.tensor(0.0, device=self.device)
            loss_temporal = torch.tensor(0.0, device=self.device)
            
            # Spatial Regularization
            if reguV > 0 or reguQU > 0:
                 # Reshape to (Nt, Ny, Nx, 3)
                 # self.params is (Nt, Ns, 3). Ns = Ny*Nx
                 params_grid = self.params.view(self.nt, self.ny, self.nx, 3)
                 
                 for t_idx in range(self.nt):
                     p_t = params_grid[t_idx] # (Ny, Nx, 3)
                     
                     if reguV > 0:
                         loss_spatial += reguV * smoothness_loss(p_t[:, :, 0], penalty='l2')
                         
                     if reguQU > 0:
                         loss_spatial += reguQU * smoothness_loss(p_t[:, :, 1], penalty='l2')
                         loss_spatial += reguQU * smoothness_loss(p_t[:, :, 2], penalty='l2')

            # Temporal Regularization
            if self.nt > 1:
                # temporal_smoothness_loss expects (Pixels, Time)
                # self.params is (Nt, Ns, 3). Permute to (Ns, Nt, 3)
                p_time = self.params.permute(1, 0, 2)
                
                if reguT_Blos > 0: loss_temporal += reguT_Blos * temporal_smoothness_loss(p_time[..., 0])
                if reguT_Bhor > 0: loss_temporal += reguT_Bhor * temporal_smoothness_loss(p_time[..., 1])
                if reguT_Bazi > 0: loss_temporal += reguT_Bazi * temporal_smoothness_loss(p_time[..., 2])

            total_loss = chi2_loss + loss_spatial + loss_temporal
            
            total_loss.backward()
            
            if self.params.grad is not None:
                 self.params.grad /= (torch.mean(torch.abs(self.params.grad)) + 1e-9)
            
            optimizer.step()
            
            t.set_postfix({'chi2': f"{chi2_loss.item():.2e}", 
                           'spatial': f"{loss_spatial.item():.2e}",
                           'temporal': f"{loss_temporal.item():.2e}"})
                
    def get_field(self) -> MagneticField:
        with torch.no_grad():
           # Flatten to N_total for MagneticField
           p_flat = self.params.reshape(-1, 3)
           
           blos_norm = p_flat[:, 0]
           bqu_packed = p_flat[:, 1:3]
           
           return MagneticField(blos_norm, bqu_packed, w_blos=self.Vnorm, w_bqu=self.QUnorm, grid_shape=self.problem.obs.grid_shape)
