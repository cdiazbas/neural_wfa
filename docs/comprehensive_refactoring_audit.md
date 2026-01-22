# Comprehensive Codebase Refactoring Report 🛠️

This document summarizes the complete reorganization and modernization of the `neural_wfa` library. The refactoring has successfully transitioned the codebase from a scattered "legacy" structure to a professional, modular Python package.

## 1. Final Architecture: `src`-layout
The project now follows the standard `src`-layout, ensuring clean imports and a clear separation of concerns.

```text
neural_wfa/
├── pyproject.toml              # Pip-installable configuration
├── README.md                   # Project documentation
├── legacy/                     # Archived legacy scripts
├── docs/                       # Project documentation & plans
├── example_py/                 # Refactored example scripts
├── tests/                      # Unit tests
├── src/
│   └── neural_wfa/
│       ├── __init__.py         # Package level API exports
│       ├── core/               # Core data structures & Physics engine
│       │   ├── magnetic_field.py   # MagneticField class (Source of Truth)
│       │   ├── observation.py      # Observation data container
│       │   └── problem.py          # WFAProblem physics & loss engine
│       ├── physics/            # Low-level physics routines
│       │   ├── lines.py            # LineInfo class
│       │   ├── derivatives.py      # cder (centered derivatives)
│       │   └── extrapolation.py    # Potential field extrapolation
│       ├── nn/                 # Neural Network components
│       │   └── architectures.py    # MLP, TemporalMLP, etc.
│       ├── optimization/       # Inversion solvers & Loss functions
│       │   ├── pixel_solver.py     # Explicit pixel-wise optimization
│       │   ├── solver.py           # Neural solver (NeuralSolver)
│       │   ├── trainers.py         # Training scheduling & helpers
│       │   └── loss.py             # Robust loss functions (Huber, Cauchy)
│       ├── regularization/     # Unified regularization schemes
│       │   ├── spatial.py          # Smoothness (L1/L2/Legacy kernels)
│       │   └── temporal.py         # Temporal smoothness & TV
│       ├── analysis/           # Post-inversion analysis
│       │   ├── uncertainty.py      # Unified uncertainty estimation
│       │   └── metrics.py          # PSNR, BPP, etc.
│       └── utils/              # System & visualization helpers
│           ├── io.py               # FITS read/write
│           ├── viz.py              # Plotting & colormaps
│           └── misc.py             # AttributeDict, scientific formatting
```

## 2. Core API Highlights

### `MagneticField`
Acts as the canonical representation of magnetic field parameters.
- **Internal Storage**: Stores normalized $(\text{Blos}, \text{BQ}, \text{BU})$ for optimization stability.
- **Physical Access**: Properties for `.blos`, `.btrans`, `.phi` (raw), `.phi_map` (visual corrected), `.inclination`.
- **Centralized Transforms**: Static/Class methods for `polar2bqu` and `bqu2polar`.
- **Format Conversion**: `.to_dict(numpy=True)` for easy analysis and plotting. Use `torch2numpy` for manual extraction.

### `Observation`
Standardizes input data handling.
- **Auto-flattening**: Handles $(H, W, 4, L)$ or $(N, 4, L)$ data seamlessly.
- **Coordinate Generation**: `.get_coordinates()` for neural field training inputs.
- **Subsetting**: `.get_pixel()` for batch processing.

### `WFAProblem`
The physics engine connecting data and models.
- **Precomputed Derivatives**: Handles `dIdw` and Doppler-scaling automatically.
- **Loss Computation**: Supports multi-stokes weighting and pixel-wise spatial weighting.
- **Batch Support**: Accepts `indices` to allow efficient mini-batch training in neural solvers.

## 3. Notable Improvements & Bug Fixes

- **Uncertainty Calibration**: Resolved the "120x discrepancy" by correctly scaling sensitivities by normalization factors within the consolidated `analysis/uncertainty.py` module.
- **Legacy Compatibility**: Restored exact 3x3 connectivity kernels in `regularization/spatial.py` to match legacy `explicit.py` behavior while adding modern L1/L2 options.
- **Solver Robustness**: `NeuralSolver` now includes potential field and azimuth regularization, gradient normalization, and learning rate scheduling.
- **Plotting**: Enforced strict layout parity with legacy code, introduced `torch2numpy` for easy tensor-to-numpy conversion, and consolidated styling in `viz.py`.
- **Formatting**: `nume2string` renamed to `format_scientific` for clarity.

## 4. Migration Summary
The following legacy files have been safely moved to the `legacy/` directory:
- `models/bfield.py` -> Ported to `core/`, `physics/`, `optimization/`.
- `models/neural_fields.py` -> Ported to `nn/`, `optimization/`.
- `models/explicit.py` -> Ported to `optimization/pixel_solver.py`.
- `models/uncertainty.py` -> Ported to `analysis/uncertainty.py`.
- `models/utils.py` -> Ported to `regularization/`, `analysis/metrics.py`, `utils/`.

## 5. Usage Example

```python
from neural_wfa.core import Observation, WFAProblem, MagneticField
from neural_wfa.physics import LineInfo
from neural_wfa.optimization import PixelSolver

# Initialize context
obs = Observation(data, wavs).to('cuda')
problem = WFAProblem(obs, LineInfo(8542))

# Run explicit inversion
solver = PixelSolver(problem)
solver.initialize_parameters(method='weak_field')
solver.solve(n_iterations=200, reguV=1e-3, reguQU=5e-2)

# Analyze results
field = solver.get_field()
blos_map = torch2numpy(field.blos_map)
# phi_map automatically handles 180-degree ambiguity [0, pi]
azi_map = torch2numpy(field.phi_map)
```

---
**Status**: Restructuring Complete ✅
**Version**: 1.0.0
**Lead Developer**: Antigravity