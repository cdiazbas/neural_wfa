import torch
import numpy as np
from typing import Optional, Union, List

class Observation:
    """
    Container for spectropolarimetric observation data.
    
    Handles:
    1. Storage of Stokes profiles (I, Q, U, V).
    2. Spectral line information (wavelengths).
    3. Masking of active spectral regions.
    4. Auto-flattening of spatial dimensions for batch processing.
    """
    
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        wavelengths: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[List[int], torch.Tensor]] = None,
        device: str = "cpu"
    ):
        """
        Args:
            data: Shape (..., 4, n_lambda). Expected to be [I, Q, U, V].
            wavelengths: Shape (n_lambda,). Angstroms relative to line center.
            mask: Indices of wavelengths to use for inversion.
            device: 'cpu' or 'cuda'.
        """
        # Convert to Tensor and Device
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if isinstance(wavelengths, np.ndarray):
            wavelengths = torch.from_numpy(wavelengths).float()
            
        self.data = data.to(device)
        self.wavelengths = wavelengths.to(device)
        self.device = device
        
        # Determine spatial/temporal dimensions
        self.grid_shape = self.data.shape[:-2] # Everything before (4, n_lambda)
        self.n_stokes, self.n_lambda = self.data.shape[-2:]
        
        # Automatic nt detection:
        # If grid_shape is (nt, ny, nx), nt is grid_shape[0]
        # If grid_shape is (ny, nx), nt is 1
        if len(self.grid_shape) == 3:
            self.nt = self.grid_shape[0]
            self.ny = self.grid_shape[1]
            self.nx = self.grid_shape[2]
        elif len(self.grid_shape) == 2:
            self.nt = 1
            self.ny = self.grid_shape[0]
            self.nx = self.grid_shape[1]
        else:
            self.nt = 1
            self.ny = self.n_pixels
            self.nx = 1
        
        # Flatten spatial dimensions for easier processing (N_pixels, 4, n_lambda)
        self.flat_data = self.data.reshape(-1, self.n_stokes, self.n_lambda)
        self.n_pixels = self.flat_data.shape[0]
        
        # Handle Mask
        if mask is not None:
            self.mask = torch.tensor(mask, device=device) if not isinstance(mask, torch.Tensor) else mask.to(device)
        else:
            self.mask = torch.arange(self.n_lambda, device=device)
            
    @property
    def stokes_I(self):
        return self.flat_data[:, 0, :]
        
    @property
    def stokes_Q(self):
        return self.flat_data[:, 1, :]
        
    @property
    def stokes_U(self):
        return self.flat_data[:, 2, :]
        
    @property
    def stokes_V(self):
        return self.flat_data[:, 3, :]
        
    @property
    def masked_wavelengths(self):
        return self.wavelengths[self.mask]
    
    def to(self, device):
        """Move data to device."""
        if self.mask is not None:
            mask_list = self.mask.tolist()
        else:
            mask_list = None
            
        return Observation(
            self.data.to(device),
            self.wavelengths.to(device),
            mask_list,
            device
        )

    def get_pixel(self, idx: Union[int, slice, torch.Tensor]) -> 'Observation':
        """Return a subset of pixels as a new Observation."""
        # This is tricky because __init__ expects spatial shape.
        # But we can reconstruct.
        subset_data = self.flat_data[idx]
        return Observation(subset_data, self.wavelengths, self.mask, self.device)

    def get_coordinates(self) -> torch.Tensor:
        """
        Generates normalized coordinates in range [-1, 1] for each pixel.
        Returns tensor of shape (N_pixels, 2) [y, x].
        """
        if len(self.grid_shape) != 2:
            return torch.linspace(-1, 1, self.n_pixels, device=self.device).unsqueeze(-1)
            
        ny, nx = self.grid_shape
        y = torch.linspace(-1, 1, ny, device=self.device)
        x = torch.linspace(-1, 1, nx, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
