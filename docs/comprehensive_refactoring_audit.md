# Comprehensive Codebase Refactoring Audit 🕵️‍♂️

## 1. Executive Summary
The current codebase is functional but structurally chaotic, making maintenance and extension difficult. It exhibits "God Object" patterns (e.g., `bfield.py`), mixed concerns (IO, physics, and plotting in the same files), and lacks standard Python packaging protocols.

## 2. Structural Issues & Inconsistencies

### A. Directory Structure
-   **Current**: `models/` is used as the source root, which is unconventional. `models` usually implies data objects or ML architectures, not the entire library logic.
-   **Missing Packaging**: No `pyproject.toml` or `setup.py`. The project is not pip-installable. Users must manually append paths to `sys.path`.

### B. File-Specific Confounding

#### `models/bfield.py` (The "God Object")
-   **Physics**: `line` class, `WFA_model3D` class.
-   **Loss Functions**: `huber_loss`, `cauchy_loss`.
-   **Legacy/Extrapolation**: `potential_extrapolation`, `embed_potential_extrapolation`, `make_square`. These belong in a separate module (e.g., `physics.extrapolation`).
-   **Conversions**: `bqu2polar`, `polar2bqu`, `bqu2polar_` (note the trailing underscore inconsistency) scatter the file.

#### `models/neural_fields.py`
-   **Typos**: `nume2string` (should be `num2string` or `format_scientific`).
-   **Mixed Responsibility**: Contains both Model definitions (`mlp`, `temporal_mlp`) and Trainer loops (`Trainer_gpu`). Ideally, `training` logic should be separate from `architecture` definitions.
-   **Plotting**: `plot_loss` is embedded here but should be in `utils.visualization`.

#### `models/utils.py`
-   **Misplaced Regularization**: Lines 234-431 contain physics-based or image-processing based regularizers (`regu`, `regu2`, `regu_mean`). These are **NOT** utilities; they are core methods for the explicit or neural optimization and should be in `regularization/` or `loss/`.
-   **Plotting**: `add_colorbar`, `torch2plot`.

#### `models/explicit.py`
-   Contains `spatial_regularization` and `temporal_regularization`. These should likely be unified with the ones found in `utils.py`.

## 3. Training & Optimization Ambiguities
-   **Blos vs params**: The codebase oscillates between optimizing raw `params` tensors and physically significant `Blos`.
-   **Trainer Complexity**: `Trainer_gpu` takes a massive list of arguments, many of which are effectively hyperparameters or configuration. A `TrainingConfig` dataclass would be cleaner.

## 4. Proposed New Architecture: `src/neural_wfa`

We propose moving to a `src`-layout structure to enforce cleaner imports and separation of concerns.

```text
neural_wfa/
├── pyproject.toml              # Standard installer
├── src/
│   └── neural_wfa/
│       ├── __init__.py
│       ├── core/               # Core data structures
│       │   ├── magnetic_field.py   # PROPOSED: New MagneticField class
│       │   └── wfa.py              # WFA_model3D, line data
│       ├── nn/                 # Neural Network components
│       │   ├── architectures.py    # mlp, temporal_mlp
│       │   └── functional.py       # Custom layers if any
│       ├── optimization/       # Training loops & Logic
│       │   ├── trainers.py         # Trainer_gpu
│       │   └── loss.py             # huber_loss, cauchy_loss
│       ├── regularization/     # Unified regularization logic
│       │   ├── spatial.py          # regu, spatial_regularization
│       │   └── temporal.py         # temporal_regularization
│       ├── physics/            # Pure physics/math helpers
│       │   ├── extrapolation.py    # potential_extrapolation
│       │   └── derivatives.py      # cder
│       └── utils/              # True utilities
│           ├── io.py               # FITS reading/writing
│           ├── viz.py              # plot_loss, add_colorbar
│           └── misc.py             # gpu selection, formatting
```

## 5. Renaming & Cleanup Todo List
-   [ ] **Rename** `nume2string` -> `format_scientific`.
-   [ ] **Move** `regu*` functions from `utils.py` to `src/neural_wfa/regularization/spatial.py`.
-   [ ] **Move** `potential_extrapolation` logic to `src/neural_wfa/physics/extrapolation.py`.
-   [ ] **Standardize** coordinate conversions (removed in favor of `MagneticField` class methods).
-   [ ] **Create** `pyproject.toml` to make package installable.

## 6. Next Steps
1.  **Safety First**: We should implement the `MagneticField` class first (as planned) as it solves the "Scattered Conversions" issue immediately.
2.  **Scaffolding**: Create the `src/neural_wfa` structure and `pyproject.toml`.
3.  **Migration**: Move files one by one, updating imports in tests/examples.


Neural WFA Library: Complete Refactoring DocumentationVersion: 2.1Date: December 2025Objective: Consolidate the neural_wfa codebase into a maintainable, pip-installable package with a unified API for both Explicit and Neural Field inversions.1. Executive SummaryThis document contains the complete source code and structural plan to refactor the neural_wfa library. The new architecture resolves the "120x Uncertainty" bug, unifies data handling via the MagneticField class, and provides a consistent API for solvers.Core ImprovementsSource of Truth: MagneticField class handles all physical unit conversions and normalizations automatically.Unified Physics: WFAProblem connects data and physics, ensuring the Forward Model is identical for Optimization and Uncertainty analysis.Bug Fix: The uncertainty module now correctly scales Jacobians by normalization factors and uses consistent Sum-of-Squares logic.Developer Experience: Solvers (PixelSolver, NeuralSolver) handle boilerplate reshaping and looping.2. Package Configurationpyproject.tomlThis file makes the library installable via pip install -e ..[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "neural_wfa"
version = "0.1.0"
description = "Weak Field Approximation inversion using Neural Fields and Explicit Optimization"
dependencies = [
    "numpy",
    "torch>=2.0",
    "astropy",
    "matplotlib",
    "scipy",
    "einops",
    "tqdm"
]

[project.optional-dependencies]
dev = ["pytest", "black", "flake8"]
3. Core Module (src/neural_wfa/core/)src/neural_wfa/__init__.py__version__ = "0.1.0"

from .core.magnetic_field import MagneticField
from .core.observation import Observation
from .core.problem import WFAProblem
from .physics.lines import LineInfo
src/neural_wfa/core/magnetic_field.pyThe "Source of Truth" for magnetic field data.import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union

class MagneticField:
    """
    Canonical representation of the Magnetic Field.
    
    Internally stores parameters in the normalized (optimization-friendly) space:
        params[..., 0] -> Blos (Normalized)
        params[..., 1] -> BQ (Normalized)
        params[..., 2] -> BU (Normalized)
        
    Provides properties to access physical units (Gauss, Radians) on the fly.
    """
    def __init__(
        self, 
        params: torch.Tensor, 
        normalization: Dict[str, float] = {'V': 1000.0, 'QU': 1e6},
        grid_shape: Optional[Tuple[int, int]] = None
    ):
        self.params = params
        self.norms = normalization
        
        # Infer shape if params is 3D (ny, nx, 3)
        if grid_shape is None and params.ndim == 3:
            self.grid_shape = params.shape[:2]
        else:
            self.grid_shape = grid_shape

    @property
    def device(self):
        return self.params.device

    def to(self, device):
        self.params = self.params.to(device)
        return self

    def detach(self):
        return MagneticField(self.params.detach(), self.norms, self.grid_shape)

    # --- Physical Properties ---

    @property
    def blos(self) -> torch.Tensor:
        """Line-of-sight magnetic field in Gauss."""
        return self.params[..., 0] * self.norms['V']

    @property
    def btrans(self) -> torch.Tensor:
        """Transverse magnetic field strength in Gauss."""
        bq = self.params[..., 1] * self.norms['QU']
        bu = self.params[..., 2] * self.norms['QU']
        return torch.sqrt(torch.sqrt(bq**2 + bu**2 + 1e-12))

    @property
    def phi(self) -> torch.Tensor:
        """Azimuthal angle in radians [0, pi]."""
        bq = self.params[..., 1]
        bu = self.params[..., 2]
        phi = 0.5 * torch.arctan2(bu, bq)
        phi = torch.where(phi < 0, phi + torch.pi, phi)
        return phi

    @property
    def inclination(self) -> torch.Tensor:
        """Inclination angle in radians."""
        return torch.arctan2(self.btrans, torch.abs(self.blos))

    # --- Regularization Logic ---

    def smoothness_loss(self, weight_blos: float = 1.0, weight_trans: float = 1.0) -> torch.Tensor:
        """Computes spatial smoothness loss (L2 of gradients)."""
        if self.grid_shape is None:
            if self.params.ndim != 3:
                raise ValueError("Grid shape required for spatial regularization.")
            field_map = self.params
        else:
            field_map = self.params.view(self.grid_shape[0], self.grid_shape[1], 3)
            
        loss = torch.tensor(0.0, device=self.device)
        
        def spatial_diff(x):
            dy = x[1:, :] - x[:-1, :]
            dx = x[:, 1:] - x[:, :-1]
            return torch.sum(dy**2) + torch.sum(dx**2)

        if weight_blos > 0:
            loss += weight_blos * spatial_diff(field_map[..., 0])
        if weight_trans > 0:
            loss += weight_trans * spatial_diff(field_map[..., 1])
            loss += weight_trans * spatial_diff(field_map[..., 2])
            
        return loss

    # --- IO Helpers ---

    def as_numpy(self) -> Dict[str, np.ndarray]:
        return {
            'blos': self.blos.detach().cpu().numpy(),
            'btrans': self.btrans.detach().cpu().numpy(),
            'phi': self.phi.detach().cpu().numpy()
        }
src/neural_wfa/core/observation.pyHandles input data reshaping.import torch
import numpy as np
from einops import rearrange

class Observation:
    """Standardizes input data to (N_pixels, 4, N_wav)."""
    def __init__(self, data, wavelength, mask=None):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data.astype(np.float32))
        if not isinstance(wavelength, torch.Tensor):
            wavelength = torch.from_numpy(wavelength.astype(np.float32))
            
        self.wavelength = wavelength
        
        # Auto-detect shape
        if data.ndim == 4: # (ny, nx, s, w)
            self.grid_shape = data.shape[:2]
            self.flat_data = rearrange(data, 'y x s w -> (y x) s w')
        elif data.ndim == 3: # (N, s, w)
            self.flat_data = data
            self.grid_shape = None
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        self.mask = mask if mask is not None else slice(None)

    def to(self, device):
        self.flat_data = self.flat_data.to(device)
        self.wavelength = self.wavelength.to(device)
        return self
src/neural_wfa/core/problem.pyThe Physics Engine.import torch
from typing import Dict, Optional, Union
from .magnetic_field import MagneticField
from .observation import Observation
from ..physics.lines import LineInfo
from ..physics.derivatives import cder

class WFAProblem:
    """
    Encapsulates the Weak Field Approximation physics problem.
    """
    def __init__(
        self, 
        observation: Observation, 
        line_info: LineInfo, 
        normalization: Dict[str, float] = {'V': 1000.0, 'QU': 1e6},
        device: str = 'cpu'
    ):
        self.obs = observation
        self.line = line_info
        self.norms = normalization
        self.device = torch.device(device)
        self.obs.to(self.device)
        
        # --- Precompute Physics Derivatives ---
        stokes_I = self.obs.flat_data[:, 0, :]
        self.dIdw = cder(self.obs.wavelength, stokes_I)
        
        wl = self.obs.wavelength
        vdop = 0.035 
        scl = 1.0 / (wl + 1e-9)
        scl[torch.abs(wl) <= vdop] = 0.0
        self.dIdwscl = self.dIdw * scl[None, :]
        
        # Physics Constants
        self.C = -4.67e-13 * (self.line.cw ** 2)
        self.C_trans = 0.75 * (self.C**2) * self.line.Gg
        self.C_long = self.C * self.line.geff

    def forward(self, field: MagneticField, batch_idx: Optional[Union[slice, torch.Tensor]] = None):
        """Returns (stokesQ, stokesU, stokesV)."""
        if field.device != self.device:
            field.to(self.device)
        if batch_idx is None:
            batch_idx = slice(None)
            
        dIdw_batch = self.dIdw[batch_idx]
        dIdwscl_batch = self.dIdwscl[batch_idx]
        
        # Use normalized params * normalization factor explicitly
        Blos_phys = field.params[batch_idx, 0, None] * self.norms['V']
        stokesV = self.C_long * Blos_phys * dIdw_batch
        
        BQ_phys = field.params[batch_idx, 1, None] * self.norms['QU']
        BU_phys = field.params[batch_idx, 2, None] * self.norms['QU']
        
        stokesQ = self.C_trans * BQ_phys * dIdwscl_batch
        stokesU = self.C_trans * BU_phys * dIdwscl_batch
        
        return stokesQ, stokesU, stokesV

    def compute_chi2(self, field: MagneticField, batch_idx=None) -> torch.Tensor:
        """Computes Total Sum of Squared Errors (Chi2)."""
        if batch_idx is None: batch_idx = slice(None)
        pred_Q, pred_U, pred_V = self.forward(field, batch_idx)
        
        obs = self.obs.flat_data[batch_idx]
        mask = self.obs.mask
        
        chi2 = torch.sum((obs[:, 1, mask] - pred_Q[..., mask])**2) + \
               torch.sum((obs[:, 2, mask] - pred_U[..., mask])**2) + \
               torch.sum((obs[:, 3, mask] - pred_V[..., mask])**2)
        return chi2
4. Physics Module (src/neural_wfa/physics/)src/neural_wfa/physics/lines.pyclass LineInfo:
    def __init__(self, cw: float = 8542):
        self.cw = float(cw)
        self.larm = 4.668645048281451e-13
        
        if cw == 8542:
            self.j1, self.j2 = 2.5, 1.5
            self.g1, self.g2 = 1.2, 1.33
        # ... (Other lines omitted for brevity) ...
        else:
            self.j1, self.j2 = 2.5, 1.5
            self.g1, self.g2 = 1.2, 1.33

        # Calculate effective Landé factor
        j1, j2, g1, g2 = self.j1, self.j2, self.g1, self.g2
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0)
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d
        
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0)
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0)
        gd = g1 - g2
        self.Gg = (self.geff * self.geff) - (
            0.0125 * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0)
        )
src/neural_wfa/physics/derivatives.pyimport torch

def cder(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes centered derivatives for non-equidistant grids."""
    if not isinstance(x, torch.Tensor): x = torch.tensor(x)
    if not isinstance(y, torch.Tensor): y = torch.tensor(y)
        
    nlam = x.shape[0]
    yp = torch.zeros_like(y)
    
    # Left
    odx = x[1] - x[0]
    yp[..., 0] = (y[..., 1] - y[..., 0]) / odx

    # Interior
    for ii in range(1, nlam - 1):
        dx = x[ii + 1] - x[ii]
        prev_diff = (y[..., ii] - y[..., ii-1]) / odx
        next_diff = (y[..., ii+1] - y[..., ii]) / dx
        yp[..., ii] = (odx * next_diff + dx * prev_diff) / (dx + odx)
        odx = dx

    # Right
    yp[..., -1] = (y[..., -1] - y[..., -2]) / (x[-1] - x[-2])
    return yp
5. Optimization Module (src/neural_wfa/optimization/)src/neural_wfa/optimization/solvers.pyThis module unifies the explicit and neural training loops.import torch
from tqdm import tqdm
from neural_wfa.core.problem import WFAProblem
from neural_wfa.core.magnetic_field import MagneticField

class PixelSolver:
    """Explicit solver for pixel-wise optimization with spatial regularization."""
    def __init__(self, problem: WFAProblem):
        self.problem = problem

    def solve(self, n_iterations=200, lr=0.1, regularization={}, initial_guess=None):
        n_pixels = self.problem.obs.flat_data.shape[0]
        
        if initial_guess is None:
            # Zero initialization
            params = torch.zeros((n_pixels, 3), requires_grad=True, device=self.problem.device)
        else:
            params = initial_guess.clone().detach().to(self.problem.device).requires_grad_(True)
            
        optimizer = torch.optim.Adam([params], lr=lr)
        
        pbar = tqdm(range(n_iterations))
        for i in pbar:
            optimizer.zero_grad()
            
            # Wrap params in MagneticField for loss calculation
            field = MagneticField(params, self.problem.norms, self.problem.obs.grid_shape)
            
            # Fidelity Loss
            loss = self.problem.compute_chi2(field)
            
            # Regularization
            reg_loss = torch.tensor(0.0, device=self.problem.device)
            if 'smoothness' in regularization:
                weight = regularization['smoothness']
                if weight > 0:
                    reg_loss += weight * field.smoothness_loss()
            
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'chi2': f"{loss.item():.2e}", 'reg': f"{reg_loss.item():.2e}"})
            
        return field.detach()


class NeuralSolver:
    """Implicit solver using Neural Fields."""
    def __init__(self, problem: WFAProblem, network: torch.nn.Module):
        self.problem = problem
        self.network = network.to(problem.device)

    def train(self, coordinates, epochs=1000, batch_size=2048, lr=1e-3, scheduler_patience=50):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)
        
        coords = coordinates.to(self.problem.device)
        n_pixels = coords.shape[0]
        indices = torch.arange(n_pixels, device=self.problem.device)
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            # Simple epoch-based batching
            perm = torch.randperm(n_pixels)
            
            epoch_loss = 0.0
            
            for i in range(0, n_pixels, batch_size):
                batch_idx = perm[i : i + batch_size]
                batch_coords = coords[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward Pass
                preds = self.network(batch_coords)
                
                # Wrap in Field
                # Note: We don't pass grid_shape here as batches are unstructured
                field = MagneticField(preds, self.problem.norms)
                
                loss = self.problem.compute_chi2(field, batch_idx=batch_idx)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            scheduler.step(epoch_loss)
            pbar.set_postfix({'loss': f"{epoch_loss:.2e}", 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})
            
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("Learning rate too small. Stopping.")
                break

        # Final Inference (Full Frame)
        with torch.no_grad():
            final_params = self.network(coords)
            
        return MagneticField(final_params, self.problem.norms, self.problem.obs.grid_shape)
6. Analysis Module (src/neural_wfa/analysis/)src/neural_wfa/analysis/uncertainty.pyThis module fixes the 120x discrepancy by ensuring Jacobians are scaled by normalization constants and using Sum-of-Squares consistency.import torch
import numpy as np
from neural_wfa.core.problem import WFAProblem
from neural_wfa.core.magnetic_field import MagneticField

def estimate_uncertainties(problem: WFAProblem, field: MagneticField):
    """
    Estimates uncertainties using Error Propagation (Hessian approximation).
    Fixes the '120x bug' by explicitly handling normalization scaling.
    """
    if field.device != problem.device:
        field.to(problem.device)

    # 1. Compute Residual Sum of Squares (RSS)
    # Using the exact same function as optimization ensures consistency
    chi2_val = problem.compute_chi2(field).item()
    
    # Degrees of Freedom: (N_pixels * N_wavelengths * 4_stokes) - N_params
    # Approximate per-pixel dof
    n_wav = problem.obs.mask_indices.stop if isinstance(problem.obs.mask, slice) else len(problem.obs.mask)
    # We have 3 params per pixel
    dof_per_pixel = (3 * n_wav) - 3 
    if dof_per_pixel < 1: dof_per_pixel = 1
    
    # 2. Compute Sensitivities (Diagonal of Jacobian^T * Jacobian)
    # J = dStokes / dParam_norm
    # We need J_phys = dStokes / dParam_phys
    # Relation: dStokes/dParam_norm = (dStokes/dParam_phys) * (dParam_phys/dParam_norm)
    #                               = (dStokes/dParam_phys) * Norm_factor
    
    # Sensitivity Blos (Stokes V)
    # Model: V = C_long * Blos_phys * dIdw
    # Jacobian_phys = C_long * dIdw
    # Jacobian_norm = C_long * dIdw * Vnorm
    J_V_norm = problem.C_long * problem.dIdw * problem.norms['V']
    
    # Sensitivity Transverse (Stokes Q, U)
    # Model: Q = C_trans * BQ_phys * dIdwscl
    # Jacobian_norm = C_trans * dIdwscl * QUnorm
    J_QU_norm = problem.C_trans * problem.dIdwscl * problem.norms['QU']
    
    # Sum of Squares of Jacobian (Diagonal Approximation of Hessian)
    # Sum over wavelengths (dim=1)
    # Shape: (N_pixels,)
    H_Blos = torch.sum(J_V_norm[:, problem.obs.mask]**2, dim=1)
    H_BQ   = torch.sum(J_QU_norm[:, problem.obs.mask]**2, dim=1)
    H_BU   = torch.sum(J_QU_norm[:, problem.obs.mask]**2, dim=1) # Same sensitivity for U
    
    # 3. Standard Error
    # Sigma^2 = (RSS / N_dof_total) * (1 / Hessian)
    # Assuming reduced chi2 is 1 (perfect fit), sigma = 1/sqrt(H)
    # If we want to scale by actual residual noise:
    
    # Pixel-wise Residuals for better estimation
    pred_Q, pred_U, pred_V = problem.forward(field)
    obs = problem.obs.flat_data
    mask = problem.obs.mask
    
    rss_V = torch.sum((obs[:, 3, mask] - pred_V[..., mask])**2, dim=1)
    rss_Q = torch.sum((obs[:, 1, mask] - pred_Q[..., mask])**2, dim=1)
    rss_U = torch.sum((obs[:, 2, mask] - pred_U[..., mask])**2, dim=1)
    
    sigma_Blos_norm = torch.sqrt(rss_V / n_wav) / torch.sqrt(H_Blos + 1e-12)
    sigma_BQ_norm   = torch.sqrt(rss_Q / n_wav) / torch.sqrt(H_BQ + 1e-12)
    sigma_BU_norm   = torch.sqrt(rss_U / n_wav) / torch.sqrt(H_BU + 1e-12)
    
    # 4. Propagate to Physical Units
    # Sigma_phys = Sigma_norm * Norm_factor (Because P_phys = P_norm * Norm)
    # Wait, check units:
    # Param_phys = Param_norm * Norm
    # var(Param_phys) = var(Param_norm) * Norm^2
    # sigma_phys = sigma_norm * Norm
    
    sigma_Blos_phys = sigma_Blos_norm * problem.norms['V']
    sigma_BQ_phys   = sigma_BQ_norm * problem.norms['QU']
    sigma_BU_phys   = sigma_BU_norm * problem.norms['QU']
    
    # 5. Propagate to Polar (Btrans, Phi)
    # Using the field's current values
    BQ = field.params[..., 1] * problem.norms['QU']
    BU = field.params[..., 2] * problem.norms['QU']
    Btrans = field.btrans
    Btrans_sq = Btrans**2
    
    # Derivatives for propagation
    # dBtrans / dBQ = BQ / (2 * Btrans)  (approx for WFA Btrans definition)
    # Note: WFA Btrans is actually defined s.t. BQ^2+BU^2 = Btrans^4 usually? 
    # Let's use the definition in MagneticField class: Btrans = (BQ^2+BU^2)^(1/4)
    # y = (x^2 + z^2)^(1/4)
    # dy/dx = 0.25 * (x^2+z^2)^(-3/4) * 2x = 0.5 * x / (Btrans^3)
    
    dBt_dBQ = 0.5 * BQ / (Btrans**3 + 1e-9)
    dBt_dBU = 0.5 * BU / (Btrans**3 + 1e-9)
    
    sigma_Btrans = torch.sqrt((dBt_dBQ * sigma_BQ_phys)**2 + (dBt_dBU * sigma_BU_phys)**2)
    
    # Phi = 0.5 * atan2(BU, BQ)
    # dPhi/dBQ = 0.5 * (-BU / (BQ^2 + BU^2))
    denom = BQ**2 + BU**2
    dPhi_dBQ = -0.5 * BU / (denom + 1e-9)
    dPhi_dBU =  0.5 * BQ / (denom + 1e-9)
    
    sigma_Phi = torch.sqrt((dPhi_dBQ * sigma_BQ_phys)**2 + (dPhi_dBU * sigma_BU_phys)**2)
    
    # Return a new MagneticField-like structure or dict
    return {
        'blos': sigma_Blos_phys.detach().cpu().numpy().reshape(problem.obs.grid_shape),
        'btrans': sigma_Btrans.detach().cpu().numpy().reshape(problem.obs.grid_shape),
        'phi': sigma_Phi.detach().cpu().numpy().reshape(problem.obs.grid_shape)
    }
7. Usage ExamplesExplicit Optimization Examplefrom neural_wfa.core import WFAProblem, Observation
from neural_wfa.physics import LineInfo
from neural_wfa.optimization import PixelSolver
import matplotlib.pyplot as plt

# 1. Setup
# obs handles reshaping (ny, nx, 4, nw) -> (N, 4, nw) automatically
obs = Observation(stokes_data, wav_grid) 
problem = WFAProblem(obs, LineInfo(8542))

# 2. Solve
solver = PixelSolver(problem)
field = solver.solve(n_iterations=200, regularization={'smoothness': 1e-3})

# 3. Visualize
# Access physical properties directly (no more manual sqrt calculations)
plt.imshow(field.btrans.cpu().numpy()) 
plt.imshow(field.phi.cpu().numpy())
Neural Field Optimization Examplefrom neural_wfa.nn import TemporalMLP # Assumed implemented similarly to old code
from neural_wfa.optimization import NeuralSolver

# 1. Setup
network = TemporalMLP(input_dim=3)
solver = NeuralSolver(problem, network) # Reuses 'problem' object

# 2. Train
# Returns a MagneticField object wrapping the network output
field_neural = solver.train(coordinates, epochs=1000)

# 3. Visualize
plt.imshow(field_neural.btrans.cpu().numpy())

I could copy some fo the ideas in the magnetic field class to the obs class to make it more consistent, as obs will also need different formats for optimization and visualization.