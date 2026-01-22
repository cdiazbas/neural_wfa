#!/usr/bin/env python
# coding: utf-8

# # Neural WFA Inversion (Refactored)
# 
# This notebook demonstrates the usage of the refactored `neural_wfa` package for inverting solar spectropolarimetric data using Neural Fields.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os, sys

# Ensure src is in path if running locally
sys.path.append("src")
sys.path.append("../src")

from neural_wfa import Observation, WFAProblem, MagneticField
from neural_wfa.physics import LineInfo
from neural_wfa.nn import MLP, HashMLP
from neural_wfa.optimization import NeuralSolver
from neural_wfa.optimization import NeuralSolver
from neural_wfa.analysis.uncertainty import estimate_uncertainties_diagonal
from neural_wfa.utils.viz import set_params
from neural_wfa.utils.viz import plot_wfa_results, plot_stokes_profiles, plot_uncertainties, torch2numpy

set_params()


# ## 1. Load Data

# In[ ]:


datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/"

img = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_dat.fits", "readonly")[0].data,
    dtype="float32",
)
xl = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_wav.fits", "readonly")[0].data,
    dtype="float32",
)

print("Data shape:", img.shape)
ny, nx, ns, nw = img.shape


# ## 2. Setup Problem

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Observation
obs = Observation(img, xl, active_wav_idx=[5, 6, 7], device=str(device))

# Line Parameters
lin = LineInfo(5173)

# WFA Physics Engine
problem = WFAProblem(obs, lin, active_wav_idx=obs.active_wav_idx, device=device)


# ## 3. Initialize Neural Fields

# In[ ]:


# Coordinate Grid (normalized -1 to 1)
y = np.linspace(-1, 1, ny)
x = np.linspace(-1, 1, nx)
YY, XX = np.meshgrid(y, x, indexing='ij')
coords = np.stack([YY, XX], axis=-1).reshape(-1, 2)
coords = torch.from_numpy(coords.astype(np.float32)).to(device)

# === HASH ENCODING VERSION SELECTOR ===
# VERSION controls which improvements are active:
#   0: Baseline (bilinear interpolation + hash collisions)
#   1: Dense grids for coarse levels (eliminates collisions)
#   2: Learnable smoothing (smooth gradients via learned MLP)
#   3: Adaptive multi-scale fusion (smart level weighting)
#   4-6: Advanced (not yet implemented)
VERSION = 6

# --- Scale / Frequency Controls ---
# In Hash Encoding, "Scales" are controlled by the Grid Resolutions.
# - base_resolution: Captures global, low-frequency structure (like low sigma).
# - max_resolution:  Captures fine, high-frequency details (like high sigma).
# Increase 'max_resolution' to resolve smaller features (e.g., 4096).
# Decrease 'base_resolution' to capture broader trends (though 16 is usually good).
BASE_RES = 2**0
MAX_RES = 2**5
NUM_LEVELS = 16

# Hash Encoding Model for Blos
model_blos = HashMLP(
    dim_in=2,
    dim_out=1,
    dim_hidden=64,
    num_layers=2,
    base_resolution=BASE_RES,
    log2_hashmap_size=19,
    max_resolution=MAX_RES,
    num_levels=NUM_LEVELS,
    version=VERSION  # Pass version
)

# Hash Encoding Model for BQU
model_bqu = HashMLP(
    dim_in=2,
    dim_out=2,
    dim_hidden=64,
    num_layers=2,
    base_resolution=BASE_RES,
    log2_hashmap_size=19,
    max_resolution=MAX_RES,
    num_levels=NUM_LEVELS,
    version=VERSION  # Pass version
)


# ## 4. Train using Neural Solver

# In[ ]:


solver = NeuralSolver(
    problem=problem,
    model_blos=model_blos,
    model_bqu=model_bqu,
    coordinates=coords,
    lr=5e-3, # Higher LR for Hash Encoding
    batch_size=200000,
    device=device
)
# Update normalization to match legacy neural script defaults (1000.0)
solver.set_normalization(w_blos=1.0, w_bqu=1000.0)

print("Training Phase 1: Blos Only...")
solver.train(n_epochs=200, optimize_blos=True, optimize_bqu=False)
loss_blos = np.array(solver.loss_history)
lr_blos = np.array(solver.lr_history)
solver.loss_history = [] # Reset for next phase
solver.lr_history = []

print("Training Phase 2: BQU Only...")
solver.train(n_epochs=200, optimize_blos=False, optimize_bqu=True)
loss_bqu = np.array(solver.loss_history)
lr_bqu = np.array(solver.lr_history)


from neural_wfa.utils.viz import plot_loss

# Phase 1
plot_loss({'loss': loss_blos, 'lr': lr_blos})
plt.savefig("ref_neural_loss_blos.png", dpi=300)
plt.show()

# Phase 2
plot_loss({'loss': loss_bqu, 'lr': lr_bqu})
plt.savefig("ref_neural_loss_bqu.png", dpi=300)
plt.show()


# ## 5. Visualize Results & Analysis

# In[ ]:


final_field = solver.get_full_field()

# 1. Magnetic Field Maps (Using new convenience properties)
blos_map = torch2numpy(final_field.blos_map)
btrans_map = torch2numpy(final_field.btrans_map)
azi_map = torch2numpy(final_field.phi_map) # Correction is automatic

plot_wfa_results(blos_map, btrans_map, azi_map, save_name=f"ref_neural_results_v{VERSION}.png")

# 2. Profile Fitting Check
# Select a pixel with strong signal
py, px = 100, 100
idx = py * nx + px
indices = torch.tensor([idx], device=device)

# Compute Model Profiles
field_sub = MagneticField(
    final_field.blos[indices],
    torch.stack([final_field.b_q[indices], final_field.b_u[indices]], dim=-1)
)

stokesQ, stokesU, stokesV = problem.compute_forward_model(field_sub, indices=indices)

obs_Q = torch2numpy(obs.stokes_Q[indices]).flatten()
obs_U = torch2numpy(obs.stokes_U[indices]).flatten()
obs_V = torch2numpy(obs.stokes_V[indices]).flatten()
mod_Q = torch2numpy(stokesQ).flatten()
mod_U = torch2numpy(stokesU).flatten()
mod_V = torch2numpy(stokesV).flatten()
wav = torch2numpy(obs.wavelengths).flatten()

mask_indices = [5, 6, 7]
plot_stokes_profiles(wav, (obs_Q, obs_U, obs_V), (mod_Q, mod_U, mod_V), 
                     mask_indices=mask_indices, save_name='ref_neural_pixel_profiles.png')

# 3. Loss (Chi2) Map (Approximation using full field)
loss_val = problem.compute_loss(final_field).item()
print(f"Total Loss: {loss_val:.4e}")

# 4. Uncertainty Estimation (Analytical)
print("Estimating Uncertainties...")
sigma_blos, sigma_btrans, sigma_phi = estimate_uncertainties_diagonal(problem, final_field)

sigma_blos = sigma_blos.reshape(ny, nx)
sigma_btrans = sigma_btrans.reshape(ny, nx)
sigma_phi = sigma_phi.reshape(ny, nx)

plot_uncertainties(sigma_blos, sigma_btrans, sigma_phi, save_name="ref_neural_uncertainties.png")


# 5. Baseline WFA Comparison
print("Computing Baseline WFA (for comparison)...")
# Using PixelSolver to get WFA guess
from neural_wfa.optimization import PixelSolver
solver_wfa = PixelSolver(problem, device=device)
solver_wfa.initialize_parameters(method='weak_field')
wfa_field = solver_wfa.get_field()

wfa_blos = torch2numpy(wfa_field.blos_map)
wfa_btrans = torch2numpy(wfa_field.btrans_map)
wfa_azi = torch2numpy(wfa_field.phi_map) # Correction is automatic

plot_wfa_results(wfa_blos, wfa_btrans, wfa_azi, save_name="ref_neural_wfa_baseline.png")

