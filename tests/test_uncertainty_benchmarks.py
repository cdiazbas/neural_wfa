"""
Comparison test for all three uncertainty estimation methods:
1. Analytical (Chi-squared based)
2. Taylor (Explicit Taylor expansion)
3. PyTorch (Hessian based)

Objective: Verify if they produce consistent results.
Note: PyTorch method is computationally expensive, so we run on a small subset.
"""

import numpy as np
import torch
import sys
import time
sys.path.append("../")

from models.bfield import WFA_model3D

def create_synthetic_data(ny=20, nx=20, nw=15, noise_level=0.001):
    """Create synthetic Stokes data for testing."""
    wl = np.linspace(-0.5, 0.5, nw)
    
    # True field
    Blos = 500 * np.sin(2 * np.pi * np.arange(ny)[:, None] / ny) * np.cos(2 * np.pi * np.arange(nx)[None, :] / nx)
    Bt = 300 * np.ones((ny, nx))
    phi = np.pi / 4 * np.ones((ny, nx))
    
    data = np.zeros((ny, nx, 4, nw))
    for i in range(ny):
        for j in range(nx):
            data[i, j, 0, :] = 1.0 - 0.5 * np.exp(-wl**2 / 0.1**2)
            dIdw = np.gradient(data[i, j, 0, :], wl)
            
            C = -4.67e-13 * 8542**2
            geff = 1.2
            data[i, j, 3, :] = C * geff * Blos[i, j] * dIdw
            
            scl = 1.0 / (wl + 1e-9)
            scl[np.abs(wl) <= 0.035] = 0.0
            dIdw_scl = dIdw * scl
            Clp = 0.75 * C**2 * 1.0 * dIdw_scl
            
            BQ = Bt[i, j]**2 * np.cos(2 * phi[i, j])
            BU = Bt[i, j]**2 * np.sin(2 * phi[i, j])
            
            data[i, j, 1, :] = Clp * BQ
            data[i, j, 2, :] = Clp * BU
    
    data += noise_level * np.random.randn(*data.shape)
    return data.astype(np.float32), wl.astype(np.float32)

def run_comparison():
    print("="*80)
    print("COMPARISON OF ALL 3 UNCERTAINTY METHODS")
    print("="*80)

    # 1. Setup
    nw = 15
    print(f"Creating synthetic data (nw={nw})...")
    data, wl = create_synthetic_data(ny=20, nx=20, nw=nw, noise_level=0.001)
    
    model = WFA_model3D(data, wl, vdop=0.035, spectral_line=8542, verbose=False)
    params = model.initial_guess(inner=True, split=False)
    
    # Select small subset for PyTorch method
    # Select ALL pixels for timing benchmark
    n_sample = len(params)
    indices = np.arange(n_sample)
    params_subset = params
    # indices = np.random.choice(len(params), n_sample, replace=False)
    # params_subset = params[indices]
    
    print(f"Testing on {n_sample} random pixels...")
    
    # 2. Run Methods
    methods = ['analytical', 'taylor', 'pytorch']
    results = {}
    
    for method in methods:
        print(f"\nRunning method='{method}'...")
        start_time = time.time()
        try:
            # Note: passing index=indices to only compute for the subset
            # But params needs to be full size if index is handled internally by filling NaNs?
            # Actually model.estimate_uncertainties takes full params and optional index.
            # But wait, looking at implementation:
            # analytical/taylor use params and index to slice.
            # pytorch uses params and index.
            # However, if we pass full params, pytorch might try to compute Hessian for everything if not careful?
            # Let's check _estimate_uncertainties_pytorch implementation.
            # It iterates over range(len(params)) if index is None.
            # If index is provided, it iterates over index.
            
            # Pass params_subset. The size of params must match the size of index for the forward pass.
            unc_blos, unc_bt, unc_phi = model.estimate_uncertainties(params_subset, index=indices, method=method)
            
            # Extract values for the subset (result is full map (ny,nx,nt) or flat?)
            # The methods reshape to (ny, nx, nt).
            # We need to flatten and extract indices.
            
            unc_blos_flat = unc_blos.reshape(-1)
            unc_bt_flat = unc_bt.reshape(-1)
            unc_phi_flat = unc_phi.reshape(-1)
            
            results[method] = {
                'blos': unc_blos_flat[indices],
                'bt': unc_bt_flat[indices],
                'phi': unc_phi_flat[indices]
            }
            print(f"  Done in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 3. Compare Results
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    # We use Taylor as the baseline for conservative estimate
    base = 'taylor'
    
    print(f"{'Method':<12} | {'Blos (mean)':<12} | {'Bt (mean)':<12} | {'Phi (mean)':<12} | {'Ratio to Taylor (Bt)':<20}")
    print("-" * 80)
    
    for method in methods:
        if method not in results: continue
        res = results[method]
        
        mean_blos = np.nanmean(res['blos'])
        mean_bt = np.nanmean(res['bt'])
        mean_phi = np.nanmean(res['phi'])
        
        ratio_bt = mean_bt / (np.nanmean(results[base]['bt']) + 1e-9)
        
        print(f"{method:<12} | {mean_blos:<12.4f} | {mean_bt:<12.4f} | {mean_phi:<12.6f} | {ratio_bt:<20.4f}")

    print("-" * 80)
    
    # Check consistency
    print("\nDetailed Sample Comparison (first pixel):")
    for method in methods:
        if method not in results: continue
        print(f"  {method}: Blos={results[method]['blos'][0]:.4f}, Bt={results[method]['bt'][0]:.4f}")

if __name__ == "__main__":
    run_comparison()
