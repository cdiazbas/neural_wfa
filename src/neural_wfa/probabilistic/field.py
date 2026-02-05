"""
Probabilistic Magnetic Field representation.
"""

import torch
from typing import Optional, Tuple, Union


class ProbabilisticMagneticField:
    """
    Probabilistic representation of magnetic field.
    
    Stores mean (μ) and log-variance (log σ²) for unified parameters [Blos, Bq, Bu].
    Optionally stores full covariance via Cholesky factors for correlated parameters.
    
    Args:
        mu: Mean values, shape (Nt, Ny, Nx, 3) or (Ny, Nx, 3) or (N, 3)
        logvar: Log-variance, same shape as mu (for diagonal covariance)
        scale_tril: Optional Cholesky factor for full covariance, shape (*grid, 3, 3)
        grid_shape: Shape of the spatial/temporal grid (e.g., (Nt, Ny, Nx))
        w_blos: Normalization weight for Blos (typically 1.0)
        w_bqu: Normalization weight for Bq, Bu (typically 1000.0)
        
    Examples:
        >>> # Diagonal covariance
        >>> field = ProbabilisticMagneticField(mu, logvar)
        >>> field.blos_mean  # Mean of Blos
        >>> field.blos_std   # Std of Blos
        >>> samples = field.sample(n=10)  # Draw 10 samples
        
        >>> # Convert to deterministic
        >>> det_field = field.to_deterministic()
    """
    
    def __init__(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        scale_tril: Optional[torch.Tensor] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        w_blos: float = 1.0,
        w_bqu: float = 1000.0,
    ):
        self.mu = mu
        self.logvar = logvar
        self.scale_tril = scale_tril
        self.grid_shape = grid_shape if grid_shape is not None else mu.shape[:-1]
        self.w_blos = w_blos
        self.w_bqu = w_bqu
        
        # Store device for convenience
        self.device = mu.device
    
    # =========================================================================
    # Mean Properties (Denormalized)
    # =========================================================================
    
    @property
    def blos_mean(self) -> torch.Tensor:
        """Mean of Blos (line-of-sight field) in Gauss."""
        return self.mu[..., 0] * self.w_blos
    
    @property
    def bq_mean(self) -> torch.Tensor:
        """Mean of Bq (Stokes Q component) in Gauss."""
        return self.mu[..., 1] * self.w_bqu
    
    @property
    def bu_mean(self) -> torch.Tensor:
        """Mean of Bu (Stokes U component) in Gauss."""
        return self.mu[..., 2] * self.w_bqu
    
    @property
    def btrans_mean(self) -> torch.Tensor:
        """Mean of transverse field magnitude: Btrans = (Bq² + Bu²)^(1/4).
        
        Note: Bq and Bu are proportional to B_perp², so btrans = sqrt(sqrt(Bq² + Bu²)).
        """
        return torch.sqrt(torch.sqrt(self.bq_mean**2 + self.bu_mean**2))
    
    @property
    def phi_mean(self) -> torch.Tensor:
        """Mean of azimuth angle = 0.5 * atan2(Bu, Bq). Returns [-pi/2, pi/2]."""
        return 0.5 * torch.atan2(self.bu_mean, self.bq_mean)
    
    @property
    def phi_mean_corrected(self) -> torch.Tensor:
        """Mean of azimuth corrected to [0, pi] range for visualization."""
        phi = self.phi_mean
        return torch.where(phi < 0, phi + torch.pi, phi)
    
    # =========================================================================
    # Uncertainty Properties (Denormalized Std)
    # =========================================================================
    
    @property
    def blos_std(self) -> torch.Tensor:
        """Standard deviation of Blos."""
        return torch.exp(0.5 * self.logvar[..., 0]) * self.w_blos
    
    @property
    def bq_std(self) -> torch.Tensor:
        """Standard deviation of Bq."""
        return torch.exp(0.5 * self.logvar[..., 1]) * self.w_bqu
    
    @property
    def bu_std(self) -> torch.Tensor:
        """Standard deviation of Bu."""
        return torch.exp(0.5 * self.logvar[..., 2]) * self.w_bqu
    
    @property
    def blos_var(self) -> torch.Tensor:
        """Variance of Blos."""
        return torch.exp(self.logvar[..., 0]) * (self.w_blos ** 2)
    
    @property
    def bq_var(self) -> torch.Tensor:
        """Variance of Bq."""
        return torch.exp(self.logvar[..., 1]) * (self.w_bqu ** 2)
    
    @property
    def bu_var(self) -> torch.Tensor:
        """Variance of Bu."""
        return torch.exp(self.logvar[..., 2]) * (self.w_bqu ** 2)
    
    # =========================================================================
    # Propagated Uncertainty (Derived Quantities)
    # =========================================================================
    
    @property
    def btrans_std(self) -> torch.Tensor:
        """
        Propagated standard deviation of Btrans = (Bq² + Bu²)^(1/4).
        
        Uses linear error propagation:
        Let f = (Bq² + Bu²)^(1/4)
        ∂f/∂Bq = 0.5 * Bq / (Bq² + Bu²)^(3/4)
        ∂f/∂Bu = 0.5 * Bu / (Bq² + Bu²)^(3/4)
        
        σ²(Btrans) = (∂f/∂Bq)² σ²(Bq) + (∂f/∂Bu)² σ²(Bu)
        """
        bqu_sq = self.bq_mean**2 + self.bu_mean**2 + 1e-10
        bqu_34 = bqu_sq ** 0.75  # (Bq² + Bu²)^(3/4)
        
        df_dbq = 0.5 * self.bq_mean / bqu_34
        df_dbu = 0.5 * self.bu_mean / bqu_34
        
        var_btrans = df_dbq**2 * self.bq_var + df_dbu**2 * self.bu_var
        return torch.sqrt(torch.clamp(var_btrans, min=0))
    
    @property
    def phi_std(self) -> torch.Tensor:
        """
        Propagated standard deviation of azimuth φ = 0.5 * atan2(Bu, Bq).
        
        Uses linear error propagation:
        ∂φ/∂Bq = -0.5 * Bu / (Bq² + Bu²)
        ∂φ/∂Bu = 0.5 * Bq / (Bq² + Bu²)
        
        Returns azimuth uncertainty in radians.
        """
        bqu_sq = self.bq_mean**2 + self.bu_mean**2 + 1e-10
        dphi_dbq = -0.5 * self.bu_mean / bqu_sq
        dphi_dbu = 0.5 * self.bq_mean / bqu_sq
        var_phi = dphi_dbq**2 * self.bq_var + dphi_dbu**2 * self.bu_var
        return torch.sqrt(torch.clamp(var_phi, min=0))
    
    # =========================================================================
    # Distribution Methods
    # =========================================================================
    
    @property
    def is_full_covariance(self) -> bool:
        """Check if full covariance is available."""
        return self.scale_tril is not None
    
    def get_distribution(self) -> torch.distributions.Distribution:
        """
        Get torch.distributions object for this field.
        
        Returns:
            - MultivariateNormal if full covariance is available
            - Independent(Normal) for diagonal covariance
        """
        if self.is_full_covariance:
            return torch.distributions.MultivariateNormal(
                loc=self.mu,
                scale_tril=self.scale_tril
            )
        else:
            return torch.distributions.Independent(
                torch.distributions.Normal(
                    loc=self.mu,
                    scale=torch.exp(0.5 * self.logvar)
                ),
                reinterpreted_batch_ndims=1
            )
    
    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Draw n samples from the distribution.
        
        Args:
            n: Number of samples to draw
            
        Returns:
            Tensor of shape (n, *grid_shape, 3) with sampled values
        """
        dist = self.get_distribution()
        return dist.rsample((n,))
    
    def log_prob(self, B: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given field values.
        
        Args:
            B: Field values, shape (*grid_shape, 3) or (n, *grid_shape, 3)
            
        Returns:
            Log probability for each pixel/point
        """
        dist = self.get_distribution()
        return dist.log_prob(B)
    
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the distribution (uncertainty measure).
        
        Returns:
            Entropy per pixel
        """
        dist = self.get_distribution()
        return dist.entropy()
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_deterministic(self):
        """
        Convert to deterministic MagneticField using the mean.
        
        Returns:
            MagneticField with point estimates (mean values)
        """
        from neural_wfa.core import MagneticField
        return MagneticField(
            blos=self.mu[..., 0:1],
            bqu=self.mu[..., 1:3],
            w_blos=self.w_blos,
            w_bqu=self.w_bqu,
            grid_shape=self.grid_shape
        )
    
    def reshape(self, *shape) -> 'ProbabilisticMagneticField':
        """
        Reshape to new grid dimensions.
        
        Args:
            shape: New grid shape (e.g., (Nt, Ny, Nx))
            
        Returns:
            New ProbabilisticMagneticField with reshaped data
        """
        new_scale_tril = None
        if self.scale_tril is not None:
            new_scale_tril = self.scale_tril.reshape(*shape, 3, 3)
        
        return ProbabilisticMagneticField(
            mu=self.mu.reshape(*shape, 3),
            logvar=self.logvar.reshape(*shape, 3),
            scale_tril=new_scale_tril,
            grid_shape=shape,
            w_blos=self.w_blos,
            w_bqu=self.w_bqu
        )
    
    def to(self, device: torch.device) -> 'ProbabilisticMagneticField':
        """Move to specified device."""
        new_scale_tril = None
        if self.scale_tril is not None:
            new_scale_tril = self.scale_tril.to(device)
        
        return ProbabilisticMagneticField(
            mu=self.mu.to(device),
            logvar=self.logvar.to(device),
            scale_tril=new_scale_tril,
            grid_shape=self.grid_shape,
            w_blos=self.w_blos,
            w_bqu=self.w_bqu
        )
    
    def detach(self) -> 'ProbabilisticMagneticField':
        """Detach from computation graph."""
        new_scale_tril = None
        if self.scale_tril is not None:
            new_scale_tril = self.scale_tril.detach()
        
        return ProbabilisticMagneticField(
            mu=self.mu.detach(),
            logvar=self.logvar.detach(),
            scale_tril=new_scale_tril,
            grid_shape=self.grid_shape,
            w_blos=self.w_blos,
            w_bqu=self.w_bqu
        )
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        cov_type = "full" if self.is_full_covariance else "diagonal"
        return (
            f"ProbabilisticMagneticField("
            f"grid_shape={self.grid_shape}, "
            f"covariance={cov_type}, "
            f"device={self.device})"
        )
