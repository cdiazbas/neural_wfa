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
from neural_wfa.nn import MLP
from neural_wfa.optimization import NeuralSolver
from neural_wfa.optimization import NeuralSolver
from neural_wfa.analysis.uncertainty import estimate_uncertainties_diagonal
from neural_wfa.utils import set_params
from neural_wfa.utils.viz import plot_wfa_results, plot_stokes_profiles, plot_uncertainties

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
# WFA Physics Engine
# Pass mask from obs or explicitly
problem = WFAProblem(obs, lin, active_wav_idx=obs.active_wav_idx, device=device)


# ## 3. Initialize Neural Fields

# In[ ]:


# Coordinate Grid (normalized -1 to 1)
y = np.linspace(-1, 1, ny)
x = np.linspace(-1, 1, nx)
YY, XX = np.meshgrid(y, x, indexing='ij')
coords = np.stack([YY, XX], axis=-1).reshape(-1, 2)
coords = torch.from_numpy(coords.astype(np.float32)).to(device)

# Model for Blos (Line-of-Sight Magnetic Field)
model_blos = MLP(
    dim_in=2,
    dim_out=1,
    dim_hidden=64,
    num_resnet_blocks=2,
    fourier_features=True,
    m_freqs=512,
    sigma=40.0,
    tune_beta=False
)

# Model for BQU (Transverse Magnetic Field Components)
model_bqu = MLP(
    dim_in=2,
    dim_out=2,
    dim_hidden=64,
    num_resnet_blocks=2,
    fourier_features=True,
    m_freqs=512,
    sigma=8.0,
    tune_beta=False
)


# ## 4. Train using Neural Solver

# In[ ]:


solver = NeuralSolver(
    problem=problem,
    model_blos=model_blos,
    model_bqu=model_bqu,
    coordinates=coords,
    lr=5e-4,
    batch_size=200000,
    device=device
)
# Update normalization to match legacy neural script defaults (1000.0)
solver.set_normalization(w_blos=1.0, w_bqu=1000.0)

print("Training Phase 1: Blos Only...")
solver.train(n_epochs=400, optimize_blos=True, optimize_bqu=False)
loss_blos = np.array(solver.loss_history)
lr_blos = np.array(solver.lr_history)
solver.loss_history = [] # Reset for next phase
solver.lr_history = []

print("Training Phase 2: BQU Only...")
solver.train(n_epochs=400, optimize_blos=False, optimize_bqu=True)
loss_bqu = np.array(solver.loss_history)
lr_bqu = np.array(solver.lr_history)


# Plot Loss History (Matching Legacy Style)
# Plot Loss History (Separated)
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
blos_map = final_field.blos_map.detach().cpu().numpy()
btrans_map = final_field.btrans_map.detach().cpu().numpy()
azi_map = final_field.phi_map.detach().cpu().numpy()

# Fix azimuth range for plotting if needed (Legacy uses 0 to Pi? or standard -Pi to Pi)
# Legacy Bazi has vmax=np.pi, vmin=0
# Our phi is -pi/2 to pi/2? Or full range?
# MagneticField.phi = 0.5 * atan2(u, q). Range is -pi/2 to pi/2 (180 deg ambig).
# Legacy code: `Bazi` from `bqu2polar` is `0.5 * arctan2`.
# Legacy Plot: `vmax=np.pi, vmin=0`.
# We need to shift/wrap if we want 0-Pi.
# But let's check legacy output. If legacy output is in [0, pi], we should match.
# Legacy `bfield.py`: `phiB[phiB < 0] += np.pi` inside `initial_guess` but NOT in `bqu2polar` (static)?
# Let's check `bqu2polar` in `bfield.py`.
# It returns `0.5 * torch.arctan2`. Output is [-pi/2, pi/2].
# But the PLOT in `legacy_neuralfield.py` expects [0, pi]?
# Let's look at legacy plot again: `cmap="twilight"`.
# The legacy code doesn't do the shift in `bqu2polar`.
# BUT `legacy_neuralfield.py` calls `bqu2polar`.
# I should ensure I match the *values* first.
# If `field.phi` follows `bqu2polar` logic, it is [-pi/2, pi/2].
# I will use `field.phi` directly. If visual mismatch occurs, I will adjust.
# Actually, I will explicitly add the shift to match the visual expectation if legacy plots show 0-Pi.
# `legacy_neuralfield.py` line 161: `vmax=np.pi, vmin=0`. Use 0-Pi range.
azi_map[azi_map < 0] += np.pi

plot_wfa_results(blos_map, btrans_map, azi_map, save_name="ref_neural_results.png")

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

obs_Q = obs.stokes_Q[indices].detach().cpu().numpy().flatten()
obs_U = obs.stokes_U[indices].detach().cpu().numpy().flatten()
obs_V = obs.stokes_V[indices].detach().cpu().numpy().flatten()
mod_Q = stokesQ.detach().cpu().numpy().flatten()
mod_U = stokesU.detach().cpu().numpy().flatten()
mod_V = stokesV.detach().cpu().numpy().flatten()
mod_V = stokesV.detach().cpu().numpy().flatten()
wav = obs.wavelengths.detach().cpu().numpy()

mask_indices = [5, 6, 7]
plot_stokes_profiles(wav, (obs_Q, obs_U, obs_V), (mod_Q, mod_U, mod_V), 
                     mask_indices=mask_indices, save_name='ref_neural_pixel_profiles.png')

# 3. Loss (Chi2) Map (Approximation using full field)
# 3. Loss (Chi2) Map (Approximation using full field)
loss_val = problem.compute_loss(final_field).item()
print(f"Total Loss: {loss_val:.4e}")

# 4. Uncertainty Estimation (Analytical)
# Neural fields provide smooth solutions, but we can still estimate analytical error propagation.
print("Estimating Uncertainties...")
# 4. Uncertainty Estimation (Analytical)
# Neural fields provide smooth solutions, but we can still estimate analytical error propagation.
print("Estimating Uncertainties...")
# Returns tuple: (sigma_blos, sigma_btrans, sigma_phi)
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

wfa_blos = wfa_field.blos.detach().cpu().numpy().reshape(ny, nx)
wfa_btrans = wfa_field.btrans.detach().cpu().numpy().reshape(ny, nx) # Uses Corrected Btrans
wfa_azi = wfa_field.phi.detach().cpu().numpy().reshape(ny, nx) 
wfa_azi[wfa_azi < 0] += np.pi # Match legacy range [0, pi]

plot_wfa_results(wfa_blos, wfa_btrans, wfa_azi, save_name="ref_neural_wfa_baseline.png")

