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
obs = Observation(img, xl, mask=[5, 6, 7], device=str(device))

# Line Parameters
lin = LineInfo(5173)

# WFA Physics Engine
# WFA Physics Engine
# Pass mask from obs or explicitly
problem = WFAProblem(obs, lin, mask=obs.mask, device=device)


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

print("Training Phase 2: BQU Only...")
solver.train(n_epochs=400, optimize_blos=False, optimize_bqu=True)


# Plot Loss History (Matching Legacy Style)
plt.figure()
loss_history = np.array(solver.loss_history)
plt.plot(loss_history, alpha=0.5)

# Smoothing (10% window)
if len(loss_history) > 10:
    window = int(len(loss_history) / 10)
    from scipy.signal import savgol_filter
    savgol_loss = savgol_filter(loss_history, window, 3 if window > 3 else 1)
    plt.plot(savgol_loss, "C0-", alpha=0.8)
    plt.plot(savgol_loss, "k-", alpha=0.2)

if len(loss_history) > 1:
    output_title_latex = (
        r"${:.2e}".format(loss_history[-1]).replace("e", "\\times 10^{")
        + "}$"
    )
plt.xlabel("Iteration")
plt.ylabel("Loss")

# Exact Legacy Title Style
if len(loss_history) > 1:
    output_title_latex = (
        r"${:.2e}".format(loss_history[-1]).replace("e", "\\times 10^{")
        + "}$"
    )
    plt.title("Final loss: " + output_title_latex)
else:
    plt.title("Training Loss")

plt.minorticks_on()
plt.yscale('log')
plt.savefig("ref_neural_loss.png", dpi=300)
plt.show()


# ## 5. Visualize Results & Analysis

# In[ ]:


final_field = solver.get_full_field()

# 1. Magnetic Field Maps
blos_map = final_field.blos.detach().cpu().numpy().reshape(ny, nx)
btrans_map = final_field.btrans.detach().cpu().numpy().reshape(ny, nx)
azi_map = final_field.phi.detach().cpu().numpy().reshape(ny, nx)

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

save_name = "ref_neural_results.png"
fs = (9*1.5, 4.5*1.5)
f, ax = plt.subplots(nrows=1, ncols=3, figsize=fs)
extent = np.float64((0, nx, 0, ny))

im0 = ax[0].imshow(blos_map, vmax=800, vmin=-800, cmap="RdGy", interpolation="nearest", extent=extent)
im1 = ax[1].imshow(btrans_map, vmin=0, vmax=800, cmap="gray", interpolation="nearest", extent=extent)
im2 = ax[2].imshow(azi_map, vmax=np.pi, vmin=0, cmap="twilight", interpolation="nearest", extent=extent)

names = [r"B$_\parallel$", r"B$_\bot$", r"$\Phi_B$"]
from neural_wfa.utils.viz import add_colorbar
add_colorbar(im0, orientation="horizontal", label=names[0] + " [G]", pad_fraction=0.17)
add_colorbar(im1, orientation="horizontal", label=names[1] + " [G]", pad_fraction=0.17)
add_colorbar(im2, orientation="horizontal", label=names[2] + " [rad]", pad_fraction=0.17)

for ii in range(1, 3): ax[ii].set_yticklabels([])
for ii in range(3):
    ax[ii].set_xlabel("x [pixels]")
    ax[ii].minorticks_on()
ax[0].set_ylabel("y [pixels]")
plt.suptitle("Neural field WFA inversion (Refactored)", fontsize=20)
plt.savefig(save_name, dpi=300)
plt.show()

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
wav = obs.wavelengths.detach().cpu().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(131); plt.plot(wav, obs_Q, 'ok', label='Obs'); plt.plot(wav, mod_Q, '-r', label='WFA'); plt.title("Stokes Q"); plt.legend()
plt.subplot(132); plt.plot(wav, obs_U, 'ok'); plt.plot(wav, mod_U, '-r'); plt.title("Stokes U")
plt.subplot(133); plt.plot(wav, obs_V, 'ok'); plt.plot(wav, mod_V, '-r'); plt.title("Stokes V")
plt.tight_layout(); plt.show()

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

plt.figure(figsize=(10, 4))
plt.subplot(121); plt.imshow(sigma_blos, cmap='inferno', origin='lower', vmin=0, vmax=50); plt.title("Sigma Blos [G]"); plt.colorbar()
plt.subplot(122); plt.imshow(sigma_btrans, cmap='inferno', origin='lower', vmin=0, vmax=200); plt.title("Sigma Btrans [G]"); plt.colorbar()
plt.savefig("ref_neural_uncertainties.png", dpi=300)
plt.tight_layout(); plt.show()


# 5. Baseline WFA Comparison
print("Computing Baseline WFA (for comparison)...")
# Using PixelSolver to get WFA guess
from neural_wfa.optimization import PixelSolver
solver_wfa = PixelSolver(problem, nt=1, device=device)
solver_wfa.initialize_parameters(method='weak_field')
wfa_field = solver_wfa.get_field()

wfa_blos = wfa_field.blos.detach().cpu().numpy().reshape(ny, nx)
wfa_btrans = wfa_field.btrans.detach().cpu().numpy().reshape(ny, nx) # Uses Corrected Btrans
wfa_azi = wfa_field.phi.detach().cpu().numpy().reshape(ny, nx) 
wfa_azi[wfa_azi < 0] += np.pi # Match legacy range [0, pi]

plt.figure(figsize=(15, 5))
plt.suptitle("Baseline WFA (Pixel-wise)", fontsize=16)
plt.subplot(131); plt.imshow(wfa_blos, cmap='RdGy', origin='lower', vmin=-800, vmax=800); plt.title("Blos")
plt.subplot(132); plt.imshow(wfa_btrans, cmap='gray', origin='lower', vmin=0, vmax=800); plt.title("Btrans")
plt.subplot(133); plt.imshow(wfa_azi, cmap='twilight', origin='lower', vmin=0, vmax=np.pi); plt.title("Azimuth")

# Exact legacy colorbar style for baseline
from neural_wfa.utils.viz import add_colorbar
add_colorbar(plt.subplot(131).images[0], orientation="horizontal", label="Blos [G]", pad_fraction=0.17)
add_colorbar(plt.subplot(132).images[0], orientation="horizontal", label="Btrans [G]", pad_fraction=0.17)
add_colorbar(plt.subplot(133).images[0], orientation="horizontal", label="Azi [rad]", pad_fraction=0.17)

plt.savefig("ref_neural_wfa_baseline.png", dpi=300)
plt.tight_layout(); plt.show()

