import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict

class MagneticField:
    """
    Represents the solar magnetic field components for WFA inversion.
    
    Acts as the 'Source of Truth' for magnetic field data, handling:
    1. Internal storage of normalized parameters (what the network predicts).
    2. On-the-fly calculation of physical quantities (Blos, Btrans, Phi).
    3. Conversion between coordinate systems (Polar <-> Cartesian).
    4. Device management (CPU/GPU).
    """
    
    def __init__(
        self, 
        blos: torch.Tensor, 
        bqu: torch.Tensor, 
        w_blos: float = 1.0, 
        w_bqu: float = 1.0
    ):
        """
        Initialize from NORMALIZED network outputs or raw optimized parameters.
        
        Args:
            blos: Normalized longitudinal field (Stokes V driver). Shape (..., 1) or (...)
            bqu: Normalized transverse field components (Stokes Q/U driver). Shape (..., 2)
            w_blos: Normalization factor for Blos (typically ~40.0)
            w_bqu: Normalization factor for BQU (typically ~8.0)
        """
        self.w_blos = w_blos
        self.w_bqu = w_bqu
        
        # Ensure correct shapes
        if blos.dim() == bqu.dim() - 1:
            blos = blos.unsqueeze(-1)
            
        self._blos_norm = blos  # (N, 1) or (H, W, 1)
        self._bqu_norm = bqu    # (N, 2) or (H, W, 2)
        
    @property
    def device(self):
        return self._blos_norm.device
        
    @property
    def shape(self):
        return self._blos_norm.shape[:-1]
    
    # --- Physical Quantities (Computed on the fly) ---
    
    @property
    def blos(self) -> torch.Tensor:
        """Longitudinal magnetic field (Gauss)."""
        return self._blos_norm.squeeze(-1) * self.w_blos
    
    @property
    def b_q(self) -> torch.Tensor:
        """Transverse component related to Stokes Q."""
        return self._bqu_norm[..., 0] * self.w_bqu

    @property
    def b_u(self) -> torch.Tensor:
        """Transverse component related to Stokes U."""
        return self._bqu_norm[..., 1] * self.w_bqu
        
    @property
    def btrans(self) -> torch.Tensor:
        """Transverse magnetic field magnitude (Gauss). Btrans = sqrt(sqrt(B_Q^2 + B_U^2))"""
        # Stokes Q/U are proportional to B_pixel^2. 
        # So the vector (BQ, BU) has magnitude B_pixel^2.
        # We need B_pixel.
        return torch.sqrt(torch.sqrt(self.b_q**2 + self.b_u**2))
        
    @property
    def phi(self) -> torch.Tensor:
        """Azimuthal angle (radians)."""
        # Checking legacy code: sin2phi = Bu/Btrans, cos2phi = Bq/Btrans
        # So tan(2phi) = Bu/Bq -> 2phi = atan2(Bu, Bq) -> phi = 0.5 * atan2(Bu, Bq)
        return 0.5 * torch.atan2(self.b_u, self.b_q)
        
    @property
    def inclination(self) -> torch.Tensor:
        """Inclination angle (gamma) w.r.t LOS."""
        return torch.atan2(self.btrans, self.blos)

    # --- Derived Cartesian Components (Local Solar Frame) ---
    # Assuming LOS is z-axis
    
    @property
    def bz(self) -> torch.Tensor:
        """Vertical component (LOS)."""
        return self.blos
        
    @property
    def bx(self) -> torch.Tensor:
        """Horizontal component x."""
        return self.btrans * torch.cos(self.phi)
        
    @property
    def by(self) -> torch.Tensor:
        """Horizontal component y."""
        return self.btrans * torch.sin(self.phi)

    # --- Utility Methods ---

    def to(self, device):
        """Move data to specified device."""
        return MagneticField(
            self._blos_norm.to(device),
            self._bqu_norm.to(device),
            self.w_blos,
            self.w_bqu
        )
        
    def detach(self):
        """Detach from computation graph."""
        return MagneticField(
            self._blos_norm.detach(),
            self._bqu_norm.detach(),
            self.w_blos,
            self.w_bqu
        )
        
    def clone(self):
        return MagneticField(
            self._blos_norm.clone(),
            self._bqu_norm.clone(),
            self.w_blos,
            self.w_bqu
        )
        
        self._blos_norm.requires_grad_(requires_grad)
        self._bqu_norm.requires_grad_(requires_grad)
        return self

    def to_polar(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns physical (Blos, Btrans, Phi) tensors."""
        return self.blos, self.btrans, self.phi

    def to_dict(self, numpy: bool = False) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Returns a dictionary of all physical components."""
        data = {
            'blos': self.blos,
            'b_q': self.b_q,
            'b_u': self.b_u,
            'btr': self.btrans,
            'phi': self.phi,
            'inclination': self.inclination
        }
        if numpy:
            return {k: v.detach().cpu().numpy() for k, v in data.items()}
        return data

    # --- Transforms & Factories ---
    
    @staticmethod
    def polar2bqu(
        blos: torch.Tensor, 
        btrans: torch.Tensor, 
        phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert [Blos, Btrans, Phi] -> [Blos, BQ, BU] with B^2 scaling."""
        bq = btrans**2 * torch.cos(2 * phi)
        bu = btrans**2 * torch.sin(2 * phi)
        return blos, bq, bu

    @staticmethod
    def bqu2polar(
        blos: torch.Tensor, 
        bq: torch.Tensor, 
        bu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert [Blos, BQ, BU] -> [Blos, Btrans, Phi] with B^2 scaling."""
        btrans = torch.sqrt(torch.sqrt(bq**2 + bu**2))
        phi = 0.5 * torch.atan2(bu, bq)
        return blos, btrans, phi

    @classmethod
    def from_polar(
        cls,
        blos: torch.Tensor, 
        btrans: torch.Tensor, 
        phi: torch.Tensor, 
        w_blos: float = 1.0, 
        w_bqu: float = 1.0
    ) -> 'MagneticField':
        """Create MagneticField from physical Blos, Btrans, Phi."""
        blos_p, bq_p, bu_p = cls.polar2bqu(blos, btrans, phi)
        
        # Normalize and pack
        norm_blos = blos_p / w_blos
        norm_bqu = torch.stack([bq_p / w_bqu, bu_p / w_bqu], dim=-1)
        
        return cls(norm_blos, norm_bqu, w_blos, w_bqu)

    @classmethod
    def from_bqu(
        cls,
        blos: torch.Tensor, 
        bq: torch.Tensor, 
        bu: torch.Tensor, 
        w_blos: float = 1.0, 
        w_bqu: float = 1.0
    ) -> 'MagneticField':
        """Create MagneticField from physical Blos, BQ, BU."""
        norm_blos = blos / w_blos
        norm_bqu = torch.stack([bq / w_bqu, bu / w_bqu], dim=-1)
        return cls(norm_blos, norm_bqu, w_blos, w_bqu)
