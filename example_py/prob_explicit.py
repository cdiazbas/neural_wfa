#!/usr/bin/env python
# coding: utf-8

# # Probabilistic WFA Inversion (Explicit Pixel Solver)
# 
# This script demonstrates the probabilistic extension of WFA inversion,
# where we estimate **probability distributions** over magnetic field parameters
# instead of point estimates.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os, sys

# Ensure src is in path
sys.path.append("src")
sys.path.append("../src")

from neural_wfa import Observation, WFAProblem
from neural_wfa.physics import LineInfo
from neural_wfa.probabilistic import ProbabilisticPixelSolver, ProbabilisticMagneticField
from neural_wfa.utils.viz import set_params, torch2numpy

set_params()


# ## 1. Load Data

# In[ ]:


datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/"

# Load single snapshot
img = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_dat.fits", "readonly")[0].data,
    dtype="float32",
)
wav = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_wav.fits", "readonly")[0].data,
    dtype="float32",
)

print(f"Data shape: {img.shape}")  # (Ny, Nx, 4, Nw)
ny, nx, ns, nw = img.shape


# ## 2. Setup Problem

# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Observation
obs = Observation(img, wav, active_wav_idx=[5, 6, 7], device=str(device))
print(f"Observation: {obs.n_pixels} pixels, {obs.n_lambda} wavelengths")

# Line Info
lin = LineInfo(5173)

# WFA Problem
problem = WFAProblem(obs, lin, device=device)


# ## 3. Probabilistic Solver

# In[ ]:


# === MODE SELECTOR ===
# Set sigma_obs to a value for fixed noise, or None for learned noise
SIGMA_OBS = None  # Try 0.01 for fixed, None for learned
SIGMA_GRANULARITY = 'per_stokes'  # 'global', 'per_stokes', or 'full'

print(f"sigma_obs mode: {'Learned' if SIGMA_OBS is None else 'Fixed'}")

solver = ProbabilisticPixelSolver(
    problem,
    sigma_obs=SIGMA_OBS,
    sigma_obs_granularity=SIGMA_GRANULARITY,
    sigma_obs_init=0.01,
    device=device
)

# Initialize with WFA estimates
solver.initialize_parameters(method='weak_field')

print(f"Parameters shape: {solver.params.shape}")  # (Nt, Ns, 6)


# ## 4. Solve

# In[ ]:


solver.solve(
    n_iterations=200,
    lr=1e-2,
    regu_spatial_blos=0.0,
    regu_spatial_bqu=0.0,
    regu_temporal_blos=0.0,  # Single frame, no temporal
    regu_temporal_bqu=0.0,
    verbose=True
)


# ## 5. Extract Results

# In[ ]:


field = solver.get_field()

print(f"Field type: {type(field)}")
print(f"Grid shape: {field.grid_shape}")
print(f"Full covariance: {field.is_full_covariance}")


# ## 6. Visualize Mean and Uncertainty

# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Means
blos_mean = torch2numpy(field.blos_mean)
btrans_mean = torch2numpy(field.btrans_mean)
phi_mean = torch2numpy(field.phi_mean)

im0 = axes[0, 0].imshow(blos_mean, cmap='RdBu_r', vmin=-500, vmax=500)
axes[0, 0].set_title('Blos Mean [G]')
plt.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(btrans_mean, cmap='inferno', vmin=0, vmax=500)
axes[0, 1].set_title('Btrans Mean [G]')
plt.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(np.degrees(phi_mean), cmap='twilight', vmin=-90, vmax=90)
axes[0, 2].set_title('Azimuth Mean [deg]')
plt.colorbar(im2, ax=axes[0, 2])

# Row 2: Uncertainties (Standard Deviation)
blos_std = torch2numpy(field.blos_std)
bq_std = torch2numpy(field.bq_std)
bu_std = torch2numpy(field.bu_std)

im3 = axes[1, 0].imshow(blos_std, cmap='viridis', vmin=0)
axes[1, 0].set_title('Blos Std [G]')
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(bq_std, cmap='viridis', vmin=0)
axes[1, 1].set_title('Bq Std [G]')
plt.colorbar(im4, ax=axes[1, 1])

im5 = axes[1, 2].imshow(bu_std, cmap='viridis', vmin=0)
axes[1, 2].set_title('Bu Std [G]')
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.savefig("prob_pixel_results.png", dpi=300)
plt.close()

print("Saved results to prob_pixel_results.png")


# ## 7. Sampling from the Distribution

# In[ ]:


# Draw 5 samples from the posterior
samples = field.sample(n=5)  # (5, Ny, Nx, 3)
print(f"Samples shape: {samples.shape}")

# Plot Blos for each sample
fig, axes = plt.subplots(1, 6, figsize=(18, 3))

# First: Mean
axes[0].imshow(blos_mean, cmap='RdBu_r', vmin=-500, vmax=500)
axes[0].set_title('Mean')
axes[0].axis('off')

# Samples
for i in range(5):
    sample_blos = torch2numpy(samples[i, ..., 0]) * field.w_blos
    axes[i+1].imshow(sample_blos, cmap='RdBu_r', vmin=-500, vmax=500)
    axes[i+1].set_title(f'Sample {i+1}')
    axes[i+1].axis('off')

plt.suptitle('Blos: Mean and 5 Posterior Samples')
plt.tight_layout()
plt.savefig("prob_pixel_samples.png", dpi=300)
plt.close()

print("Saved samples to prob_pixel_samples.png")


# ## 8. Loss History

# In[ ]:


losses = solver.loss_history
nll = [l['nll'] for l in losses]
spatial = [l['spatial'] for l in losses]

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(nll, label='NLL')
ax.semilogy(spatial, label='Spatial Prior')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend()
ax.set_title('Probabilistic Solver Loss History')
plt.tight_layout()
plt.savefig("prob_pixel_loss.png", dpi=300)
plt.close()

print("Saved loss history to prob_pixel_loss.png")


# ## 9. Check Learned sigma_obs (if applicable)

# In[ ]:


if SIGMA_OBS is None:
    sigma_final = torch2numpy(solver.sigma_obs)
    print(f"Learned sigma_obs shape: {sigma_final.shape}")
    print(f"Learned sigma_obs (per-Stokes mean):")
    print(f"  Q: {sigma_final[0].mean():.4e}")
    print(f"  U: {sigma_final[1].mean():.4e}")
    print(f"  V: {sigma_final[2].mean():.4e}")
else:
    print(f"Fixed sigma_obs: {SIGMA_OBS}")

print("\nDone!")
