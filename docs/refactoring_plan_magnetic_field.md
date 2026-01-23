# Magnetic Field Class Refactoring Plan

## 1. Problem Statement
The current codebase handles magnetic field parameters in a fragmented manner:
- **Multiple Representations**: Data exists as `(Blos, BQ, BU)` (linear-ish parameters) or `(Blos, Btrans, Phi)` (polar).
- **Inconsistent Shapes**: Functions expect sometimes `(N, 3)`, sometimes `(ny, nx, 3)`, sometimes separate arrays.
- **Normalization Confusion**: `Vnorm` and `QUnorm` are applied manually in various places, leading to potential bugs if forgot.
- **Scattered Conversions**: Helper functions like `bqu2polar`, `polar2bqu`, `make_square` are scattered in `bfield.py`.
- **Backend Mixing**: A mix of PyTorch tensors and NumPy arrays requires explicit `.cpu().numpy()` conversions.

## 2. Proposed Solution: `MagneticField` Class
We will introduce a unified class `MagneticField` that acts as the canonical data structure for all magnetic field operations.

### Core Philosophy
- **Canonical Storage**: Internally store `(Blos, BQ, BU)` as PyTorch tensors. This is the "native" format for the WFA linear inversion.
- **On-the-fly Views**: Properties like `.btrans`, `.phi`, `.inclination` are computed on demand (and cached if necessary, though lightweight usually).
- **Normalization Aware**: The class knows its normalization factors (`Vnorm`, `QUnorm`) and can export "physical" or "normalized" values.
- **Backend Agnostic API**: seamless export to NumPy or keeping in PyTorch.

## 3. Design Specification

### Class Signature
```python
class MagneticField:
    def __init__(self, data, coordinates='bqu', normals=(1.0, 1000.0), shape=None):
        """
        data: Tensor or Array.
        coordinates: 'bqu' (Blos, BQ, BU) or 'polar' (Blos, Btrans, Phi)
        normals: (Vnorm, QUnorm). If provided, data is assumed NORMALIZED.
                 Or maybe data is always assumed physical?
                 DECISION: Store PHYSICAL values internally to avoid confusion?
                 OR Store NORMALIZED values because gradients flow through them?

                 DECISION: Store NORMALIZED values + normalization constants.
                 This allows direct use in optimization (gradients) while providing physical property accessors.
        """
        self.params = ... # (N, 3) or (ny, nx, 3) Tensor, normalized
        self.Vnorm = normals[0]
        self.QUnorm = normals[1]
        self.grid_shape = shape # (ny, nx) or None if flattened
```

### Properties (Physical Units)
- `blos`: Returns `params[..., 0] * Vnorm`
- `bq`: Returns `params[..., 1] * QUnorm`
- `bu`: Returns `params[..., 2] * QUnorm`
- `btrans`: Returns `(bq^2 + bu^2)^0.25` (Wait, definition of Btrans in WFA is tricky. `Btrans_wfa = (Bt_model)^?`. WFA uses `Btrans` such that linear polarization is proportional to `Btrans^2`?)
    - *Correction*: In WFA code, `Btrans` usually refers to the actual transverse field strength.
    - Check code: `Btr = (BQ^2 + BU^2)^(1/4)` -> This implies `BQ, BU` are NOT linear polarization $Q, U$.
    - Actually `BQ \propto Btrans^2 * cos(2phi)`, `BU \propto Btrans^2 * sin(2phi)`.
    - So canonical parameters `BQ`, `BU` in the code are actually "Linearly Polarizing Field Components" or similar.
    - The class should abstract this ambiguity.
- `phi`: Returns `0.5 * atan2(BU, BQ)` (Raw azimuth within `[-pi/2, pi/2]`).
- `phi_map`: Returns azimuth corrected to `[0, pi]` for visualization (resolving 180-degree ambiguity).
- `vec`: Returns `(Bx, By, Bz)` assuming some geometry? Or just `(Blos, Btrans, Phi)`.

### Methods
- `to(device)`: Moves internal tensors.
- `reshape(ny, nx)`: Reshapes flat list to grid.
- `flatten()`: Flattens grid to list.
- `as_numpy()`: Returns dict of numpy arrays `{'blos': ..., 'btrans': ...}`.
- `visualize()`: Quick helper to plot maps (delegates to plotting utils).

## 4. Workflows Review

### A. Forward Model (`WFA_model3D`)
- **Current**: `forward(params)` where `params` is `(N, 3)` tensor.
- **New**: `forward(field: MagneticField)`?
  - Or `forward(field.normalized_params)`.
  - Keeping `forward` accepting tensors is good for `torch.optim` which optimizes a *Tensor*, not a class.
  - **Refined Plan**: The *Optimizer* works on a Tensor property of the class (or just a Tensor). The Class is a wrapper for *interpretability* and *IO*.
  - So `WFA_model3D` might still take tensors internally, but high-level API uses `MagneticField`.

### B. Uncertainty
- The uncertainty module returns 3 separate arrays.
- **New**: Return a `MagneticField` instance containing the uncertainties?
  - Or `MagneticFieldUncertainty` class?
  - Or simply `MagneticField` initialized with sigma values. (Though physical meaning of BQ_sigma is distinct).

### C. Neural Fields
- Neural Network outputs `(N, 3)` tensor.
- We wrap this output into `MagneticField(output_tensor, normals=...)`.
- Then we can easily query `field.btrans` to log/monitor training.

## 5. Implementation Roadmap

1.  **Phase 1: Foundation**:
    - Create `models/magnetic_field.py`.
    - Implement `MagneticField` class with unit tests.
    - Ensure all coordinate conversions from `bfield.py` are ported as methods.

2.  **Phase 2: Integration**:
    - Update `WFA_model3D` to optionally accept `MagneticField` object or use it for initialization.
    - Refactor `initial_guess` to return `MagneticField`.

3.  **Phase 3: Cleanup**:
    - Deprecate standalone conversion functions in `bfield.py`.
    - Update example notebooks to demonstrate the new "User Friendly" API.

4.  **Phase 4: API Refinement (Completed)**:
    - Implemented `torch2numpy` in `viz.py` to standardize data extraction.
    - Added `phi_map` auto-correction to `MagneticField` to simplify visualization scripts.
    - Enforced strict plotting layout parity with legacy code.

## 6. Brainstorming Questions (Self-Correction)
- **Q**: Should it handle `Bz` (vertical) vs `Blos` (line-of-sight)?
- **A**: WFA is strictly LoS. But `potential_extrapolation` uses `Bz`. We can add a method `assume_vertical()` which aliases `blos` to `bz`.

- **Q**: How to handle pixel-wise vs neural field?
- **A**: The class doesn't care how `data` was generated. It just holds values. If `data` is `(N, 3)`, it's a list (valid for pixel-wise before reshaping). If `(ny, nx, 3)`, it's a map. `reshape()` handles transitions.

- **Q**: Gradients?
- **A**: If `data` requires grad, accessing properties like `.btrans` will propagate gradients correctly through PyTorch operations. This is powerful for regularization (e.g. "minimize Btrans gradient").
