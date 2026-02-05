"""
Probabilistic Pixel Solver for WFA Inversion.

Optimizes probability distributions over magnetic field parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from typing import Optional, Union

from neural_wfa.core.problem import WFAProblem
from neural_wfa.probabilistic.field import ProbabilisticMagneticField
from neural_wfa.probabilistic.utils import (
    broadcast_sigma_obs, 
    create_learnable_sigma_obs,
    log_sigma_to_sigma
)
from neural_wfa.regularization.spatial import smoothness_loss
from neural_wfa.regularization.temporal import temporal_smoothness_loss


class ProbabilisticPixelSolver:
    """
    Probabilistic Solver for Explicit (Pixel-wise) WFA Inversion.
    
    Optimizes probability distributions (mean + variance) for unified magnetic
    field parameters [Blos, Bq, Bu] at each pixel.
    
    Args:
        problem: WFAProblem with observation and physics
        sigma_obs: Observation noise. Can be:
            - None: Learn as free parameter
            - float: Global fixed noise
            - Tensor (3,): Per-Stokes fixed noise
            - Tensor (3, Nw): Full fixed noise
        sigma_obs_granularity: If sigma_obs is None, granularity for learned noise:
            'global', 'per_stokes', or 'full'
        sigma_obs_init: Initial value for learned sigma_obs
        device: Computation device
        
    Examples:
        >>> solver = ProbabilisticPixelSolver(problem, sigma_obs=0.01)  # Fixed
        >>> solver = ProbabilisticPixelSolver(problem, sigma_obs=None)  # Learned
        >>> solver.initialize_parameters()
        >>> solver.solve(n_iterations=200)
        >>> field = solver.get_field()  # ProbabilisticMagneticField
    """
    
    def __init__(
        self,
        problem: WFAProblem,
        sigma_obs: Optional[Union[float, torch.Tensor]] = None,
        sigma_obs_granularity: str = 'global',  # 'global', 'per_stokes', or 'full'
        sigma_obs_init: float = 0.01,
        n_samples: int = 1,  # Number of samples for reparameterization
        device: torch.device = None
    ):
        self.problem = problem
        self.nt = problem.nt
        self.device = device if device else problem.device
        
        # Determine spatial dimensions
        obs_shape = problem.obs.grid_shape
        if len(obs_shape) == 3:  # (Nt, Ny, Nx)
            self.ny = obs_shape[1]
            self.nx = obs_shape[2]
            self.n_spatial = self.ny * self.nx
        elif len(obs_shape) == 2:  # (Ny, Nx)
            self.ny = obs_shape[0]
            self.nx = obs_shape[1]
            self.n_spatial = self.ny * self.nx
        else:
            self.n_spatial = problem.obs.n_pixels // self.nt
            self.ny = self.n_spatial
            self.nx = 1
        
        self.n_pixels = problem.obs.n_pixels
        self.n_wavelengths = problem.obs.n_lambda
        
        # =====================================================================
        # Parameters: Unified [Blos, Bq, Bu] with diagonal covariance
        # Shape: (Nt, Ns, 6) where last dim is [μ_blos, μ_bq, μ_bu, logσ²_blos, logσ²_bq, logσ²_bu]
        # =====================================================================
        self.params = nn.Parameter(
            torch.zeros(self.nt, self.n_spatial, 6, device=self.device)
        )
        
        # Normalization constants (same as PixelSolver)
        self.w_blos = 1000.0     # Blos normalization (Vnorm)
        self.w_bqu = 1e6         # Bq, Bu normalization (QUnorm)
        
        # Scaling Factors for LogVar (Preconditioning)
        self.logvar_scale_blos = 1.0
        self.logvar_scale_bqu = 1.0
        
        # Number of samples for reparameterization trick
        self.n_samples = n_samples
        
        # Initial log-variance (corresponds to σ ~ 0.03 in normalized space => ~30G)
        self.init_logvar = -9.0
        
        # Dual Optimization Parameter: Alpha (Temperature)
        # Initialize alpha ~ 0.01 directly
        self.alpha = nn.Parameter(torch.tensor(0.01, device=self.device))
        
        # =====================================================================
        # Observation Noise: Fixed or Learned
        # =====================================================================
        self.sigma_obs_fixed = sigma_obs is not None
        
        if self.sigma_obs_fixed:
            # Fixed observation noise
            sigma_broadcasted = broadcast_sigma_obs(
                sigma_obs, n_stokes=3, n_wavelengths=self.n_wavelengths, device=self.device
            )
            self.register_buffer_or_attr('_sigma_obs', sigma_broadcasted)
            self._log_sigma_obs = None
        else:
            # Learnable observation noise
            self._log_sigma_obs = create_learnable_sigma_obs(
                granularity=sigma_obs_granularity,
                n_stokes=3,
                n_wavelengths=self.n_wavelengths,
                init_value=sigma_obs_init,
                device=self.device
            )
            self._sigma_obs = None
        
        self.sigma_obs_granularity = sigma_obs_granularity
        
        # Loss history
        self.loss_history = []
    
    def register_buffer_or_attr(self, name: str, tensor: torch.Tensor):
        """Register as buffer (for nn.Module) or simple attribute."""
        setattr(self, name, tensor)
    
    @property
    def sigma_obs(self) -> torch.Tensor:
        """Get observation noise as (3, Nw) tensor."""
        if self.sigma_obs_fixed:
            return self._sigma_obs
        else:
            return log_sigma_to_sigma(
                self._log_sigma_obs,
                n_stokes=3,
                n_wavelengths=self.n_wavelengths,
                device=self.device
            )
    
    def get_all_parameters(self):
        """Get list of all optimizable parameters."""
        params = [self.params]
        if not self.sigma_obs_fixed:
            params.append(self._log_sigma_obs)
        return params


    # =========================================================================
    # Initialization
    # =========================================================================
    
    def initialize_parameters(self, method: str = 'weak_field'):
        """
        Initialize mean parameters.
        
        Initialize parameters using weak field approximation or random values.
        """
        with torch.no_grad():
            if method == 'weak_field':
                # Use WFA for mean initialization
                Blos, BQ, BU = self.problem.compute_initial_guess()
                
                # Error handling for reshape: just use view_as
                blos_norm = (Blos / self.w_blos).view_as(self.params[..., 0])
                bqu_norm_q = (BQ / self.w_bqu).view_as(self.params[..., 1])
                bqu_norm_u = (BU / self.w_bqu).view_as(self.params[..., 2])
                
                # Assign to parameters
                self.params[..., 0] = blos_norm
                self.params[..., 1] = bqu_norm_q
                self.params[..., 2] = bqu_norm_u
                
                # Set log-variances (scaled)
                # target_v = -7.0. param = -7.0 / 10.0 = -0.7
                self.params[..., 3] = self.init_logvar / self.logvar_scale_blos
                self.params[..., 4:6] = self.init_logvar / self.logvar_scale_bqu
                
            else:
                # Random initialization
                self.params.normal_(0, 0.1)
                self.params[..., 3] = self.init_logvar / self.logvar_scale_blos
                self.params[..., 4:6] = self.init_logvar / self.logvar_scale_bqu
    
    # =========================================================================
    # Forward Model
    # =========================================================================
    
    def forward_model(self, B: torch.Tensor) -> torch.Tensor:
        """
        Forward model: Magnetic Field -> Stokes Profiles using WFA logic (or full synthesis).
        Here we use the analytical proxy derived in WFAProblem.
        
        Args:
            B: Magnetic field vector (N, 3) [Blos, BQ, BU]
        
        Returns:
            Predicted Stokes (N, 3, Nw) for Q, U, V
        """
        Blos = B[:, 0]  # (N,)
        BQ = B[:, 1]
        BU = B[:, 2]
        
        # Physics constants
        C = self.problem.C
        geff = self.problem.lin.geff
        Gg = self.problem.lin.Gg
        
        # Derivatives (N, Nw)
        dIdw = self.problem.dIdw
        dIdwscl = self.problem.dIdwscl
        
        # Forward model
        stokesV = C * geff * Blos.unsqueeze(-1) * dIdw
        Clp = 0.75 * (C**2) * Gg * dIdwscl
        stokesQ = Clp * BQ.unsqueeze(-1)
        stokesU = Clp * BU.unsqueeze(-1)
        
        return torch.stack([stokesQ, stokesU, stokesV], dim=1)  # (N, 3, Nw)
    
    # =========================================================================
    # Loss Computation
    # =========================================================================
    
    def compute_nll(
        self,
        y_obs: torch.Tensor,
        y_pred: torch.Tensor,
        sigma_obs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute Gaussian Negative Log-Likelihood.
        
        Args:
            y_obs: Observed Stokes (N, 3, Nw)
            y_pred: Predicted Stokes (N, 3, Nw)
            sigma_obs: Observation noise (3, Nw)
            mask: Wavelength mask (optional)
            
        Returns:
            Scalar NLL loss
        """
        # Residual
        residual = y_obs - y_pred
        
        # Apply mask if provided
        if mask is not None:
            residual = residual[..., mask]
            sigma_obs = sigma_obs[..., mask]
        
        # Gaussian NLL: 0.5 * (residual²/σ² + log(σ²))
        # = 0.5 * (residual²/σ² + 2*log(σ))
        var_obs = sigma_obs ** 2
        nll = 0.5 * (residual**2 / var_obs + torch.log(var_obs))
        
        return nll.sum()
    
    # =========================================================================
    # Solve
    # =========================================================================
    
    def solve(
        self,
        n_iterations: int = 200,
        lr: float = 1e-2,
        optimizer_cls: type = optim.Adam,
        optimizer_kwargs: dict = None,
        # Spatial regularization (prior on smoothness)
        regu_spatial_blos: float = 1e-3,
        regu_spatial_bqu: float = 0.5e-1,
        # Temporal regularization (prior on temporal smoothness)
        regu_temporal_blos: float = 1e-3,
        regu_temporal_bqu: float = 1e-3,
        # Variance regularization
        min_logvar: float = -10.0,  # Minimum log-variance (σ_min ~ 0.007)

        target_noise: float = 0.0121, # Target NLL (derived from sigma_obs ~ 0.0121)
        lr_alpha: float = 1e-4,       # Learning rate for dual parameter alpha (Reduced for extensive loss scaling)
        verbose: bool = True
    ):
        """
        Run probabilistic optimization with Maximum-Entropy Constraint.
        
        Objective: Minimize L = NLL - alpha * Entropy
        Constraint: NLL <= Target (enforced by adaptive alpha)
        
        Args:
            n_iterations: Number of optimization steps
            lr: Learning rate for parameters
            optimizer_cls: Optimizer class
            optimizer_kwargs: Extra optimizer arguments
            regu_spatial_*: Spatial smoothness weights
            regu_temporal_*: Temporal smoothness weights
            min_logvar: Minimum log-variance
            target_noise: Expected observation noise level (sigma_obs)
            lr_alpha: Learning rate for alpha adjustment
            verbose: Show progress bar
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        optimizer = optimizer_cls(self.get_all_parameters(), lr=lr, **optimizer_kwargs)
        
        # Dual Alpha Optimizer (simple gradient ascent on alpha)
        # We update alpha manually, so no optimizer needed for it.
        
        hook_handle = None
        
        t = trange(n_iterations, leave=True) if verbose else range(n_iterations)
        
        for i in t:
            optimizer.zero_grad()
            
            # =================================================================
            # Extract parameters and compute variance
            # =================================================================
            mu = self.params[..., :3]  # (Nt, Ns, 3)
            logvar_params = self.params[..., 3:6]  # (Nt, Ns, 3)
            
            # Apply scaling to get real log-variance
            logvar_blos = logvar_params[..., 0:1] * self.logvar_scale_blos
            logvar_bqu = logvar_params[..., 1:3] * self.logvar_scale_bqu
            logvar = torch.cat([logvar_blos, logvar_bqu], dim=-1)
            
            # Clamp log-variance to prevent collapse
            logvar = torch.clamp(logvar, min=min_logvar)
            std = torch.exp(0.5 * logvar)  # (Nt, Ns, 3)
            
            # =================================================================
            # Monte Carlo NLL using torch.distributions
            # =================================================================
            # Observed Stokes (computed once, used for all samples)
            obs_Q = self.problem.obs.stokes_Q  # (N, Nw)
            obs_U = self.problem.obs.stokes_U
            obs_V = self.problem.obs.stokes_V
            stokes_obs = torch.stack([obs_Q, obs_U, obs_V], dim=1)  # (N, 3, Nw)
            mask = self.problem.active_wav_idx
            
            # Create distribution
            dist = torch.distributions.Normal(mu, std)
            
            nll_loss = 0
            for k in range(self.n_samples):
                # Sample using reparameterization trick (via rsample)
                B_norm = dist.rsample()  # (Nt, Ns, 3)
                
                # Denormalize to physical units
                B_denorm = torch.stack([
                    B_norm[..., 0] * self.w_blos,
                    B_norm[..., 1] * self.w_bqu,
                    B_norm[..., 2] * self.w_bqu
                ], dim=-1).reshape(-1, 3)
                
                # Forward model
                stokes_pred = self.forward_model(B_denorm)  # (N, 3, Nw)
                
                # NLL for this sample
                nll_loss += self.compute_nll(stokes_obs, stokes_pred, self.sigma_obs, mask)
            
            nll_loss = nll_loss / self.n_samples  # Average over samples
            
            # =================================================================
            # Prior Loss (Spatial Smoothness)
            # =================================================================
            loss_spatial = torch.tensor(0.0, device=self.device)
            
            # Reshape mu for spatial smoothness: (Nt, Ny, Nx, 3)
            mu_spatial = mu.view(self.nt, self.ny, self.nx, 3)
            
            if regu_spatial_blos > 0:
                loss_spatial += regu_spatial_blos * smoothness_loss(mu_spatial[..., 0], penalty='l2')
            if regu_spatial_bqu > 0:
                loss_spatial += regu_spatial_bqu * smoothness_loss(mu_spatial[..., 1], penalty='l2')
                loss_spatial += regu_spatial_bqu * smoothness_loss(mu_spatial[..., 2], penalty='l2')
            
            # =================================================================
            # Prior Loss (Temporal Smoothness)
            # =================================================================
            loss_temporal = torch.tensor(0.0, device=self.device)
            
            if self.nt > 1:
                if regu_temporal_blos > 0:
                    loss_temporal += regu_temporal_blos * smoothness_loss(mu_spatial[..., 0], penalty='l2_temporal')
                if regu_temporal_bqu > 0:
                    loss_temporal += regu_temporal_bqu * smoothness_loss(mu_spatial[..., 1], penalty='l2_temporal')
                    loss_temporal += regu_temporal_bqu * smoothness_loss(mu_spatial[..., 2], penalty='l2_temporal')
            
            
            # =================================================================
            # Maximum-Entropy Loss (NLL - alpha * H)
            # =================================================================
            
            # =================================================================
            # Maximum-Entropy Loss (NLL - alpha * H)
            # =================================================================
            
            # 1. Compute Entropy of Gaussian: H = 0.5 * sum(log(2*pi*e*sigma^2))
            # H prop to sum(log_var) (Extensive)
            entropy = 0.5 * logvar.sum()
            
            # 2. Get current alpha (ensure non-negative)
            alpha_val = self.alpha.item()
            
            # 3. MaxEnt Loss
            loss_maxent = nll_loss - self.alpha * entropy
            
            # 4. Variance Barrier (Soft Clamping)
            # prevent logvar from going below min_logvar
            loss_var_barrier = 10.0 * torch.relu(min_logvar - logvar).pow(2).sum()

            # =================================================================
            # Total Loss
            # =================================================================
            total_loss = loss_maxent + loss_spatial + loss_temporal + loss_var_barrier
            
            # Backward
            total_loss.backward()
            
            # =================================================================
            # Diagnostic Logging
            # =================================================================
            # Target NLL (Extensive): 0.5 * (1 + log(target^2)) * N_active_pixels * 3 * Nw
            # Note: nll_loss is summed over active pixels/wavelengths.
            # We need to count how many data points are actually active.
            
            n_active_wav = len(mask) if mask is not None else self.n_wavelengths
            # Solver optimizes Q,U,V explicitly (stacking them at line 350)
            n_stokes = 3 
            n_data_points = self.n_pixels * n_stokes * n_active_wav
            
            # Dynamic Target Update (Crucial for Learnable Sigma)
            # If sigma_obs changes, the target NLL must change to reflect the new noise floor.
            with torch.no_grad():
                current_sigma = self.sigma_obs.mean()
                target_nll_val = 0.5 * (1.0 + torch.log(current_sigma**2)) * n_data_points
                target_nll_val = target_nll_val.item()

            if verbose and i % 50 == 0:
                 print(f"\n[Iter {i}] MaxEnt State:")
                 print(f"  NLL:     {nll_loss.item():.4e} (Target: {target_nll_val:.4e})")
                 print(f"  Entropy: {entropy.item():.4e}")
                 print(f"  Alpha:   {alpha_val:.4e}")
                 print(f"  Sigma:   {std.mean().item():.4e}")
            
            optimizer.step()
            
            # =================================================================
            # Dual Update for Alpha (Linear Ascent)
            # =================================================================
            
            with torch.no_grad():
                nll_curr = nll_loss.item()
                # Update: alpha += lr * (Target - NLL)
                # If Target > NLL (Overfitting!): Alpha increases -> More Entropy.
                # If Target < NLL (Underfitting!): Alpha decreases -> Less Entropy.
                
                # Note: NLL and Target are extensive (Sum), so gradients are large.
                # Scales are ~10^5. LR needs to be small (e.g. 1e-4 or 1e-5).
                
                alpha_grad = (target_nll_val - nll_curr)
                self.alpha.data += lr_alpha * alpha_grad
                self.alpha.data.clamp_(min=0.0)
            
            # Enforce log-variance constraint? No, used barrier.
            
            # Record
            self.loss_history.append({
                'total': total_loss.item(),
                'nll': nll_loss.item(),
                'entropy': entropy.item(),
                'alpha': alpha_val,
                'spatial': loss_spatial.item(),
                'temporal': loss_temporal.item(),
                'sigma': std.mean().item(),
                'target': target_nll_val,
                'sigma_obs': self.sigma_obs.mean().item()
            })
            
            if verbose:
                std_mean = torch.exp(0.5 * self.params[..., 3:6]).mean().item()
                t.set_postfix({
                    'nll': f'{nll_loss.item():.3e}',
                    'ent': f'{entropy.item():.3e}',
                    'alp': f'{alpha_val:.3e}',
                    'std': f'{std_mean:.3e}'
                })
        
        # Cleanup hook
        if hook_handle is not None:
            hook_handle.remove()
        
        # Print final summary
        if verbose:
            print(f"\nFinal Loss: nll={nll_loss.item():.4e}, "
                  f"entropy={entropy.item():.4e}, "
                  f"alpha={alpha_val:.4e}")
            if not self.sigma_obs_fixed:
                # Access raw parameter (before broadcasting)
                if self.sigma_obs_granularity == 'per_stokes':
                    sigma_vals = torch.exp(self._log_sigma_obs).detach().cpu().numpy()
                    print(f"Learned σ_obs: Q={sigma_vals[0]:.4e}, U={sigma_vals[1]:.4e}, V={sigma_vals[2]:.4e}, mean={sigma_vals.mean():.4e}")
                else:
                    print(f"Learned σ_obs (mean): {self.sigma_obs.mean().item():.4e}")
    
    # =========================================================================
    # Get Field
    # =========================================================================
    
    def get_field(self) -> ProbabilisticMagneticField:
        """
        Get the current probabilistic magnetic field.
        
        Returns:
            ProbabilisticMagneticField with mean and variance
        """
        with torch.no_grad():
            mu = self.params[..., :3].clone()  # (Nt, Ns, 3)
            logvar = self.params[..., 3:6].clone() # (Nt, Ns, 3)
            
            # Reshape to grid
            if self.nt > 1:
                grid_shape = (self.nt, self.ny, self.nx)
            else:
                grid_shape = (self.ny, self.nx)
            
            mu_grid = mu.view(*grid_shape, 3)
            logvar_grid = logvar.view(*grid_shape, 3)
            
            return ProbabilisticMagneticField(
                mu=mu_grid,
                logvar=logvar_grid,
                grid_shape=grid_shape,
                w_blos=self.w_blos,
                w_bqu=self.w_bqu
            )
