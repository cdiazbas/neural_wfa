#!/usr/bin/env python
# coding: utf-8

# # Explicit WFA Inversion (Refactored)
# 
# This notebook demonstrates the usage of the refactored `neural_wfa` package for EXPLICIT (pixel-by-pixel) inversion.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os, sys

# Ensure src is in path if running locally
sys.path.append("src")
sys.path.append("../src")

from neural_wfa import Observation, WFAProblem
from neural_wfa.physics import LineInfo
from neural_wfa.optimization import PixelSolver
from neural_wfa.analysis.uncertainty import estimate_uncertainties_diagonal
from neural_wfa import MagneticField
from legacy.plot_params import set_params
from neural_wfa.utils.viz import plot_wfa_results, plot_stokes_profiles, plot_chi2_maps, plot_uncertainties
set_params()

# ## 1. Load Data

# In[ ]:


# Handle data path
datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/" # Fallback if running from inside dir

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


# Force CPU for explicit examples to ensure stability
device = torch.device("cpu") 
print("Using device:", device)

# Observation
obs = Observation(img, xl, active_wav_idx=[5, 6, 7], device=str(device))

# Line Parameters
lin = LineInfo(5173)

# WFA Physics Engine
problem = WFAProblem(obs, lin, weights=[10, 10, 10], active_wav_idx=torch.tensor([5, 6, 7]), device=device)


# ## 3. Pixel Solver

# In[ ]:


# Initialize Solver
solver = PixelSolver(problem, device=device)
    
# Initialize Parameters using Weak Field Approximation (Much faster convergence)
solver.initialize_parameters(method='weak_field')

# --- Plot Initial Guess (WFA) ---
print("Plotting Initial Guess (WFA)...")
initial_field = solver.get_field()
# Using new convenience properties (automatically reshaped)
blos_0 = initial_field.blos_map.detach().cpu().numpy()
btrans_0 = initial_field.btrans_map.detach().cpu().numpy()
azi_0 = initial_field.phi_map.detach().cpu().numpy()
azi_0[azi_0 < 0] += np.pi

plot_wfa_results(blos_0, btrans_0, azi_0, save_name='ref_initial_guess.png')
# -------------------------------

print("Starting Inversion...")
solver.solve(
    n_iterations=200,
    lr=1e-2, # Explicit optimization usually allows higher LR than Neural
    reguV=1e-4, 
    reguQU=1e-2,
    verbose=True
)


# ## 4. Analysis & Visualization

# In[ ]:


final_field = solver.get_field()

# 1. Magnetic Field Maps
blos_map = final_field.blos_map.detach().cpu().numpy()
btrans_map = final_field.btrans_map.detach().cpu().numpy()
azi_map = final_field.phi_map.detach().cpu().numpy()
azi_map[azi_map < 0] += np.pi

plot_wfa_results(blos_map, btrans_map, azi_map, save_name='ref_results.png')

# 2. Profile Fitting Check
# Select a pixel with strong signal (Matching Legacy: 150, 150)
px, py = 150, 150 # Legacy uses (150, 150)
x, y = 150, 150 
idx = py * nx + px
indices = torch.tensor([idx], device=device)

# Compute Model Profiles
field_sub = MagneticField(
    final_field.blos[indices],
    torch.stack([final_field.b_q[indices], final_field.b_u[indices]], dim=-1)
)

stokesQ, stokesU, stokesV = problem.compute_forward_model(field_sub, indices=indices)

obs_Q = obs.stokes_Q[indices].detach().cpu().numpy().flatten()
obs_U = obs.stokes_U[indices].detach().cpu().numpy().flatten()
obs_V = obs.stokes_V[indices].detach().cpu().numpy().flatten()
mod_Q = stokesQ.detach().cpu().numpy().flatten()
mod_U = stokesU.detach().cpu().numpy().flatten()
mod_V = stokesV.detach().cpu().numpy().flatten()
wav = obs.wavelengths.detach().cpu().numpy()

mask_indices = [5, 6, 7]
plot_stokes_profiles(wav, (obs_Q, obs_U, obs_V), (mod_Q, mod_U, mod_V), 
                     mask_indices=mask_indices, save_name='ref_pixel_profiles.png')

# 3. Loss (Chi2) Map
loss_val = problem.compute_loss(final_field).item()
print(f"Total Loss: {loss_val:.2f}")

# 3b. Chi2 Spatial Maps
print("Computing Chi2 Maps...")
stokesQ, stokesU, stokesV = problem.compute_forward_model(final_field)

obs_Q = obs.stokes_Q.detach().cpu().numpy().reshape(ny, nx, nw)
obs_U = obs.stokes_U.detach().cpu().numpy().reshape(ny, nx, nw)
obs_V = obs.stokes_V.detach().cpu().numpy().reshape(ny, nx, nw)

mod_Q = stokesQ.detach().cpu().numpy().reshape(ny, nx, nw)
mod_U = stokesU.detach().cpu().numpy().reshape(ny, nx, nw)
mod_V = stokesV.detach().cpu().numpy().reshape(ny, nx, nw)

diff_Q = mod_Q - obs_Q
sigma_est = np.std(diff_Q)
print(f"Estimated sigma: {sigma_est:.9f}")

mask_indices = [5, 6, 7] # As defined in problem setup
chi2_Q = np.mean((mod_Q[..., mask_indices] - obs_Q[..., mask_indices])**2, axis=2) / (sigma_est**2)
chi2_U = np.mean((mod_U[..., mask_indices] - obs_U[..., mask_indices])**2, axis=2) / (sigma_est**2)
chi2_V = np.mean((mod_V[..., mask_indices] - obs_V[..., mask_indices])**2, axis=2) / (sigma_est**2)
chi2_total = chi2_Q + chi2_U + chi2_V

plot_chi2_maps(chi2_Q, chi2_U, chi2_V, chi2_total=chi2_total,
               save_name_components='ref_chi2_components.png',
               save_name_total='ref_chi2_total.png')

# 4. Uncertainty Estimation (Analytical)
print("Estimating Uncertainties...")
unc_blos, unc_btrans, unc_phi = estimate_uncertainties_diagonal(problem, final_field)

unc_blos = unc_blos.reshape(ny, nx)
unc_btrans = unc_btrans.reshape(ny, nx)
unc_phi = unc_phi.reshape(ny, nx)

plot_uncertainties(unc_blos, unc_btrans, unc_phi, save_name='ref_uncertainties.png')

