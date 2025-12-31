# Uncertainty Analysis for WFA Model

## Table of Contents
1. [Forward Model Structure](#forward-model-structure)
2. [Parameter Dependencies](#parameter-dependencies)
3. [Why Hessian is Not Needed](#why-hessian-is-not-needed)
4. [Uncertainty Estimation Methods](#uncertainty-methods)
5. [Numerical Stability & Validation](#numerical-stability)

---

## Forward Model Structure

The WFA (Weak Field Approximation) model is **linear** in the fitted parameters:

```
V(λ) = C * g_eff * Blos * dI/dλ
Q(λ) = C_lp(λ) * BQ
U(λ) = C_lp(λ) * BU
```

where:
- `C = -4.67e-13 * λ₀²` (constant)
- `C_lp(λ) = 0.75 * C² * G_g * dI/dλ / λ` (wavelength-dependent sensitivity)
- `BQ = Bt² * cos(2φ)` (transverse field component)
- `BU = Bt² * sin(2φ)` (transverse field component)

**Key insight**: The model is LINEAR in (Blos, BQ, BU), which means first-order Taylor expansion is exact for these parameters.

---

## Parameter Dependencies

### Direct Parameters (Linear)
1. **Blos**: Linear dependence on Stokes V
   - `V ∝ Blos`
   - Sensitivity: `∂V/∂Blos = C * g_eff * dI/dλ`

2. **BQ, BU**: Linear dependence on Stokes Q, U
   - `Q ∝ BQ`, `U ∝ BU`
   - Sensitivity: `∂Q/∂BQ = ∂U/∂BU = C_lp(λ)`

### Derived Parameters (Non-linear)
3. **Bt, φ**: Non-linear transformation from BQ, BU
   - `Bt = (BQ² + BU²)^(1/4)`
   - `φ = 0.5 * arctan2(BU, BQ)`
   - Requires Jacobian for uncertainty propagation

---

## Why Hessian is Not Needed

The Hessian (matrix of second derivatives) is only needed for non-linear models or complex parameter correlations. For linear parameters (Blos, BQ, BU), the **Jacobian provides exact uncertainty estimation**.

**For the WFA model**:
- ✅ First-order Taylor expansion (Jacobian) is **exact**
- ✅ Computationally efficient
- ❌ Hessian adds no new information for the linear parameters

---

## Uncertainty Estimation Methods {#uncertainty-methods}

We have implemented two approaches to estimate uncertainties, which are theoretically **identical**:

### 1. Chi-squared Based (Standard)
Derives uncertainty from the curvature of the chi-squared surface.
Formula: `σ = √(χ² / (n_wavelengths * ||J||²))`

### 2. Explicit Taylor Expansion
Derives uncertainty from residual RMS and sensitivity.
Formula: `σ = RMS(residuals) / ||J||`

### Mathematical Equivalence
Since `RMS² = χ² / n_wavelengths`, the two methods are mathematically identical:
`σ_taylor = √(χ²/n) / ||J|| = √(χ² / (n * ||J||²)) = σ_chi2`

---

## Numerical Stability & Validation {#numerical-stability}

### The "Factor of 120" Paradox (RESOLVED)

During development, we observed a discrepancy where the two methods differed by a factor of ~120 for BQ/BU parameters, while agreeing for Blos.

**Root Cause**: Numerical stability with epsilon.
- The sensitivity for Q/U is extremely small (`~10⁻⁷`) because it depends on the second order of `C` (`~10⁻¹⁰`).
- The denominator in the chi-squared method involves the square of this sensitivity (`~10⁻¹⁴`).
- A standard regularization epsilon of `1e-9` was added to the denominator: `denom + 1e-9`.
- Since `1e-9 >> 10⁻¹⁴`, the result was dominated by epsilon, effectively decoupling the uncertainty from the physics.

**Fix**:
- We reduced the epsilon to `1e-20` (safe for float32/64 operations in this context) to ensure the physical denominator dominates.
- **Result**: Both methods now produce **identical results**, confirming the math.

### Validation Results (Corrected)
With the fix, both methods agree within numerical precision (< 0.1%).

### PyTorch Method
We also implemented a PyTorch-based method using `torch.autograd` to compute the Hessian.
- **Finding**: The PyTorch method computes covariance assuming unit noise variance ($\sigma=1$).
- **Correction**: To match the analytical results, the PyTorch output must be scaled by the actual residual RMS ($\sigma_{noise} \approx 10^{-3}$).
- Once scaled, it aligns with the analytical methods.

---

## Recommendations

**Use the default `analytical` method** in `WFA_model3D`.

Advantages:
1. **Correct**: Verified to match rigorous Taylor expansion.
2. **Fast**: Uses vectorized analytical formulas, no autograd overhead.
3. **Robust**: Now fixed with appropriate numerical stability constants.

For derived parameters (Bt, φ), the code automatically propagates uncertainties using the Jacobian of the transformation:
- `σ_Bt = √((∂Bt/∂BQ * σ_BQ)² + (∂Bt/∂BU * σ_BU)²)`
