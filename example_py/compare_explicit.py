#!/usr/bin/env python
# coding: utf-8

# # Comparison: Deterministic vs Probabilistic Explicit Solver
#
# This script runs both solvers and produces side-by-side comparison plots.

import torch
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os
import sys
import json

# Ensure src is in path
sys.path.append("src")
sys.path.append("../src")

from neural_wfa import Observation, WFAProblem
from neural_wfa.physics import LineInfo
from neural_wfa.optimization import PixelSolver
from neural_wfa.probabilistic import ProbabilisticPixelSolver
from neural_wfa.analysis.uncertainty import estimate_uncertainties_diagonal
from neural_wfa.utils.viz import set_params, torch2numpy

set_params()


# =============================================================================
# Configuration
# =============================================================================
# Test with and without regularization
RUN_WITH_REGULARIZATION = True
RUN_WITHOUT_REGULARIZATION = True

# Fixed sigma from deterministic residuals
SIGMA_FROM_DETERMINISTIC = 0.01211

# Set to None to LEARN sigma_obs instead of fixing it
USE_LEARNED_SIGMA = True  # Learned/dynamic observational noise to match residuals

# Regularization weights (matching deterministic)
REGU_V = 1e-4
REGU_QU = 1e-2


# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

datadir = "example_py/plage_sst/"
if not os.path.exists(datadir):
    datadir = "plage_sst/"

img = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_dat.fits", "readonly")[0].data,
    dtype="float32",
)
wav = np.ascontiguousarray(
    fits.open(datadir + "CRISP_5173_plage_wav.fits", "readonly")[0].data,
    dtype="float32",
)

print(f"Data shape: {img.shape}")
ny, nx, ns, nw = img.shape


# =============================================================================
# 2. Setup Problem
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

obs = Observation(img, wav, active_wav_idx=[5, 6, 7], device=str(device))
lin = LineInfo(5173)
problem = WFAProblem(obs, lin, device=device)


# =============================================================================
# Helper Functions
# =============================================================================
def run_deterministic(problem, ny, nx, regu_v=0.0, regu_qu=0.0, label="det"):
    """
    Run deterministic PixelSolver and return results.
    """
    print(f"\n[Deterministic {label}] regu_v={regu_v}, regu_qu={regu_qu}")

    solver = PixelSolver(problem, device=problem.device)
    solver.initialize_parameters(method="weak_field")
    solver.solve(n_iterations=200, lr=1e-2, reguV=regu_v, reguQU=regu_qu, verbose=True)

    field = solver.get_field()

    # Extract maps
    blos = torch2numpy(field.blos_map).reshape(ny, nx)
    btrans = torch2numpy(field.btrans_map).reshape(ny, nx)
    phi = torch2numpy(field.phi_map).reshape(ny, nx)

    # Analytical uncertainties
    unc_blos, unc_btrans, unc_phi = estimate_uncertainties_diagonal(problem, field)
    unc_blos = unc_blos.reshape(ny, nx)
    unc_btrans = unc_btrans.reshape(ny, nx)
    unc_phi = unc_phi.reshape(ny, nx)

    return {
        "blos": blos,
        "btrans": btrans,
        "phi": phi,
        "unc_blos": unc_blos,
        "unc_btrans": unc_btrans,
        "unc_phi": unc_phi,
        "solver": solver,
    }


def run_probabilistic(
    problem, sigma_obs=None, regu_blos=0.0, regu_bqu=0.0, label="prob", init_logvar=-5.0
):
    """
    Run probabilistic ProbabilisticPixelSolver and return results.
    """
    # MaxEnt Parameters
    TARGET_NOISE = 0.0121
    LR_ALPHA = 1e-6  # Reduced for extensive quantity scaling (Delta ~ 1e5)

    print(f"\n[Probabilistic {label}] sigma_obs={sigma_obs}, init_logvar={init_logvar}")
    print(f"MaxEnt Params: target={TARGET_NOISE}, lr_alpha={LR_ALPHA}")

    solver = ProbabilisticPixelSolver(
        problem,
        sigma_obs=sigma_obs,
        sigma_obs_granularity="per_stokes",
        sigma_obs_init=0.01,
        device=problem.device,
    )
    solver.init_logvar = init_logvar
    solver.initialize_parameters(method="weak_field")

    solver.solve(
        n_iterations=800,  # Increased for long-term evolution check
        lr=1e-2,
        regu_spatial_blos=regu_blos,
        regu_spatial_bqu=regu_bqu,
        target_noise=TARGET_NOISE,
        lr_alpha=LR_ALPHA,
        min_logvar=-10.0,
        verbose=True,
    )

    field = solver.get_field()

    # Extract maps (already shaped)
    blos = torch2numpy(field.blos_mean)
    btrans = torch2numpy(field.btrans_mean)
    # Correct phi to [0, pi] range (same as deterministic)
    phi = torch2numpy(field.phi_mean)
    phi = np.where(phi < 0, phi + np.pi, phi)

    unc_blos = torch2numpy(field.blos_std)
    unc_btrans = torch2numpy(field.btrans_std)
    unc_phi = torch2numpy(field.phi_std)

    sigma_learned = torch2numpy(solver.sigma_obs) if sigma_obs is None else None

    return {
        "blos": blos,
        "btrans": btrans,
        "phi": phi,
        "unc_blos": unc_blos,
        "unc_btrans": unc_btrans,
        "unc_phi": unc_phi,
        "sigma_learned": sigma_learned,
        "solver": solver,
        "history": solver.loss_history,
    }


def plot_dual_dynamics(
    history, target_nll_extensive, title_suffix="", save_name="dual_dynamics.png"
):
    """
    Plot the evolution of Dual Optimization variables.
    """
    iterations = range(len(history))
    nll = [h["nll"] for h in history]
    entropy = [h["entropy"] for h in history]
    alpha = [h["alpha"] for h in history]
    sigma = [h["sigma"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"MaxEnt Dynamics{title_suffix}", fontsize=16)

    # 1. NLL vs Target
    axes[0, 0].plot(iterations, nll, label="NLL")

    # Check if target is in history (dynamic)
    if "target" in history[0]:
        target = [h["target"] for h in history]
        axes[0, 0].plot(iterations, target, "r--", label="Target NLL (Dynamic)")
    else:
        axes[0, 0].axhline(
            y=target_nll_extensive, color="r", linestyle="--", label="Target NLL"
        )  # Extensive
    axes[0, 0].set_title("NLL Evolution")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Entropy
    axes[0, 1].plot(iterations, entropy, color="green")
    axes[0, 1].set_title("Entropy Evolution")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Entropy")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Alpha (Temperature)
    axes[1, 0].plot(iterations, alpha, color="orange")
    axes[1, 0].set_title("Alpha (Temperature) Evolution")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Alpha")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Mean Sigma
    axes[1, 1].plot(iterations, sigma, color="purple")
    axes[1, 1].set_title("Mean Sigma Evolution")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Sigma")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.close()
    print(f"Saved: {save_name}")


def save_history_json(history, save_name="history.json"):
    """
    Save history to JSON file.
    """
    with open(save_name, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved Log: {save_name}")


def plot_comparison(
    det_results, prob_results, title_suffix="", save_name="comparison.png"
):
    """
    Create side-by-side comparison plot.
    """

    fig, axes = plt.subplots(6, 2, figsize=(12, 24))

    # Shared colormaps and ranges
    blos_vmax = 500
    btrans_vmax = 500
    unc_blos_vmax = max(
        np.nanpercentile(det_results["unc_blos"], 95),
        np.nanpercentile(prob_results["unc_blos"], 95),
    )
    unc_btrans_vmax = max(
        np.nanpercentile(det_results["unc_btrans"], 95),
        np.nanpercentile(prob_results["unc_btrans"], 95),
    )
    unc_phi_vmax = max(
        np.nanpercentile(np.degrees(det_results["unc_phi"]), 95),
        np.nanpercentile(np.degrees(prob_results["unc_phi"]), 95),
    )

    # Row 0: Blos Mean
    im00 = axes[0, 0].imshow(
        det_results["blos"], cmap="RdBu_r", vmin=-blos_vmax, vmax=blos_vmax
    )
    axes[0, 0].set_title("Deterministic: Blos [G]")
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(
        prob_results["blos"], cmap="RdBu_r", vmin=-blos_vmax, vmax=blos_vmax
    )
    axes[0, 1].set_title("Probabilistic: Blos [G]")
    plt.colorbar(im01, ax=axes[0, 1])

    # Row 1: Btrans Mean
    im10 = axes[1, 0].imshow(
        det_results["btrans"], cmap="gray", vmin=0, vmax=btrans_vmax
    )
    axes[1, 0].set_title("Deterministic: Btrans [G]")
    plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(
        prob_results["btrans"], cmap="gray", vmin=0, vmax=btrans_vmax
    )
    axes[1, 1].set_title("Probabilistic: Btrans [G]")
    plt.colorbar(im11, ax=axes[1, 1])

    # Row 2: Azimuth Mean
    im20 = axes[2, 0].imshow(
        np.degrees(det_results["phi"]), cmap="twilight", vmin=0, vmax=180
    )
    axes[2, 0].set_title("Deterministic: Azimuth [deg]")
    plt.colorbar(im20, ax=axes[2, 0])

    im21 = axes[2, 1].imshow(
        np.degrees(prob_results["phi"]), cmap="twilight", vmin=0, vmax=180
    )
    axes[2, 1].set_title("Probabilistic: Azimuth [deg]")
    plt.colorbar(im21, ax=axes[2, 1])

    # Row 3: Blos Uncertainty
    im30 = axes[3, 0].imshow(
        det_results["unc_blos"], cmap="cividis", vmin=0, vmax=unc_blos_vmax
    )
    axes[3, 0].set_title("Deterministic: std(Blos) [G]")
    plt.colorbar(im30, ax=axes[3, 0])

    im31 = axes[3, 1].imshow(
        prob_results["unc_blos"], cmap="cividis", vmin=0, vmax=unc_blos_vmax
    )
    axes[3, 1].set_title("Probabilistic: std(Blos) [G]")
    plt.colorbar(im31, ax=axes[3, 1])

    # Row 4: Btrans Uncertainty
    im40 = axes[4, 0].imshow(
        det_results["unc_btrans"], cmap="cividis", vmin=0, vmax=unc_btrans_vmax
    )
    axes[4, 0].set_title("Deterministic: std(Btrans) [G]")
    plt.colorbar(im40, ax=axes[4, 0])

    im41 = axes[4, 1].imshow(
        prob_results["unc_btrans"], cmap="cividis", vmin=0, vmax=unc_btrans_vmax
    )
    axes[4, 1].set_title("Probabilistic: std(Btrans) [G]")
    plt.colorbar(im41, ax=axes[4, 1])

    # Row 5: Azimuth Uncertainty
    im50 = axes[5, 0].imshow(
        np.degrees(det_results["unc_phi"]), cmap="cividis", vmin=0, vmax=unc_phi_vmax
    )
    axes[5, 0].set_title("Deterministic: std(phi) [deg]")
    plt.colorbar(im50, ax=axes[5, 0])

    im51 = axes[5, 1].imshow(
        np.degrees(prob_results["unc_phi"]), cmap="cividis", vmin=0, vmax=unc_phi_vmax
    )
    axes[5, 1].set_title("Probabilistic: std(phi) [deg]")
    plt.colorbar(im51, ax=axes[5, 1])

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(
        f"Deterministic vs Probabilistic Comparison{title_suffix}", fontsize=14, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_name}")


def print_metrics(det_results, prob_results, label=""):
    """
    Print quantitative comparison metrics.
    """
    print(f"\n{'=' * 60}")
    print(f"QUANTITATIVE COMPARISON {label}")
    print(f"{'=' * 60}")

    # Mean differences
    diff_blos = np.abs(det_results["blos"] - prob_results["blos"])
    diff_btrans = np.abs(det_results["btrans"] - prob_results["btrans"])

    print(f"Mean Difference (Blos):   {np.nanmean(diff_blos):.2f} G")
    print(f"Mean Difference (Btrans): {np.nanmean(diff_btrans):.2f} G")

    # Detailed Sigma Statistics
    print("\nSigma Statistics (Blos):")
    print(
        f"  Det : Min={np.nanmin(det_results['unc_blos']):.2f}, Max={np.nanmax(det_results['unc_blos']):.2f}, Mean={np.nanmean(det_results['unc_blos']):.2f}, Median={np.nanmedian(det_results['unc_blos']):.2f}"
    )
    print(
        f"  Prob: Min={np.nanmin(prob_results['unc_blos']):.2f}, Max={np.nanmax(prob_results['unc_blos']):.2f}, Mean={np.nanmean(prob_results['unc_blos']):.2f}, Median={np.nanmedian(prob_results['unc_blos']):.2f}"
    )

    print("\nSigma Statistics (Btrans):")
    print(
        f"  Det : Min={np.nanmin(det_results['unc_btrans']):.2f}, Max={np.nanmax(det_results['unc_btrans']):.2f}, Mean={np.nanmean(det_results['unc_btrans']):.2f}, Median={np.nanmedian(det_results['unc_btrans']):.2f}"
    )
    print(
        f"  Prob: Min={np.nanmin(prob_results['unc_btrans']):.2f}, Max={np.nanmax(prob_results['unc_btrans']):.2f}, Mean={np.nanmean(prob_results['unc_btrans']):.2f}, Median={np.nanmedian(prob_results['unc_btrans']):.2f}"
    )

    # Uncertainty ratios
    ratio_blos = prob_results["unc_blos"] / (det_results["unc_blos"] + 1e-10)
    ratio_btrans = prob_results["unc_btrans"] / (det_results["unc_btrans"] + 1e-10)

    print(f"Median σ Ratio (Blos):   {np.nanmedian(ratio_blos):.3f}")
    print(f"Median σ Ratio (Btrans): {np.nanmedian(ratio_btrans):.3f}")

    # Correlation of uncertainties
    corr_blos = np.corrcoef(
        det_results["unc_blos"].flatten(), prob_results["unc_blos"].flatten()
    )[0, 1]
    corr_btrans = np.corrcoef(
        det_results["unc_btrans"].flatten(), prob_results["unc_btrans"].flatten()
    )[0, 1]

    print(f"Correlation σ(Blos):   {corr_blos:.4f}")
    print(f"Correlation σ(Btrans): {corr_btrans:.4f}")

    # Coverage (% of det within prob ± 2σ)
    within_2sigma_blos = (
        np.abs(det_results["blos"] - prob_results["blos"])
        < 2 * prob_results["unc_blos"]
    )
    within_2sigma_btrans = (
        np.abs(det_results["btrans"] - prob_results["btrans"])
        < 2 * prob_results["unc_btrans"]
    )

    print(f"Coverage (Blos within 2σ):   {100 * np.mean(within_2sigma_blos):.1f}%")
    print(f"Coverage (Btrans within 2σ): {100 * np.mean(within_2sigma_btrans):.1f}%")


def check_noise_calibration(solver, problem, sigma_obs_used, label=""):
    """
    Validate sigma_obs by computing actual residuals std(Stokes_syn - Stokes_obs).
    This checks if the assumed observational noise matches the actual fit residuals.
    """
    print(f"\n{'=' * 60}")
    print(f"OBSERVATIONAL NOISE CALIBRATION CHECK {label}")
    print(f"{'=' * 60}")

    with torch.no_grad():
        # Get final parameters
        if hasattr(solver, "params"):  # Probabilistic solver
            mu = solver.params[..., :3]
            # Denormalize
            blos = (mu[..., 0] * solver.w_blos).reshape(-1)
            bq = (mu[..., 1] * solver.w_bqu).reshape(-1)
            bu = (mu[..., 2] * solver.w_bqu).reshape(-1)
            B = torch.stack([blos, bq, bu], dim=-1)
        else:  # Deterministic solver
            p_flat = solver.params.reshape(-1, 3)
            blos = p_flat[:, 0] * solver.Vnorm
            bq = p_flat[:, 1] * solver.QUnorm
            bu = p_flat[:, 2] * solver.QUnorm
            B = torch.stack([blos, bq, bu], dim=-1)

        # Forward model to get synthetic Stokes
        stokes_syn = solver.forward_model(B)  # (N, 3, Nw)

        # Observed Stokes
        stokes_obs = torch.stack(
            [problem.obs.stokes_Q, problem.obs.stokes_U, problem.obs.stokes_V], dim=1
        )  # (N, 3, Nw)

        # Compute residuals
        residuals = stokes_syn - stokes_obs  # (N, 3, Nw)

        # Use only active wavelengths
        mask = problem.active_wav_idx
        residuals_active = residuals[:, :, mask]  # (N, 3, Nmask)

        # Compute std for each Stokes component
        std_Q = residuals_active[:, 0, :].std().item()
        std_U = residuals_active[:, 1, :].std().item()
        std_V = residuals_active[:, 2, :].std().item()
        std_all = residuals_active.std().item()

        # Per-pixel std (across wavelengths)
        std_per_pixel = residuals_active.std(dim=2).mean(dim=1)  # (N,)
        std_per_pixel_mean = std_per_pixel.mean().item()
        std_per_pixel_median = std_per_pixel.median().item()

    print("\nAssumed σ_obs:")
    if isinstance(sigma_obs_used, (float, int)):
        print(f"  {sigma_obs_used:.6f} (scalar)")
        sigma_Q = sigma_U = sigma_V = sigma_obs_used
    else:
        # Array (per-Stokes)
        sigma_obs_array = np.array(sigma_obs_used).flatten()
        if len(sigma_obs_array) == 3:
            sigma_Q, sigma_U, sigma_V = sigma_obs_array
            print(f"  Q: {sigma_Q:.6f}")
            print(f"  U: {sigma_U:.6f}")
            print(f"  V: {sigma_V:.6f}")
        else:
            sigma_Q = sigma_U = sigma_V = sigma_obs_array[0]
            print(f"  {sigma_obs_array[0]:.6f} (from array)")

    print("\nActual Residual Statistics:")
    print(f"  std(Q residuals): {std_Q:.6f}")
    print(f"  std(U residuals): {std_U:.6f}")
    print(f"  std(V residuals): {std_V:.6f}")
    print(f"  std(All):         {std_all:.6f}")

    print("\nPer-Pixel Residual std:")
    print(f"  Mean:   {std_per_pixel_mean:.6f}")
    print(f"  Median: {std_per_pixel_median:.6f}")

    # Calibration ratio (use per-component if available)
    if isinstance(sigma_obs_used, (float, int)):
        calibration_ratio = std_all / sigma_obs_used
        print(f"\nCalibration Ratio (actual/assumed): {calibration_ratio:.3f}")
    else:
        ratio_Q = std_Q / sigma_Q
        ratio_U = std_U / sigma_U
        ratio_V = std_V / sigma_V
        calibration_ratio = std_all / np.mean([sigma_Q, sigma_U, sigma_V])
        print("\nCalibration Ratios (actual/assumed):")
        print(f"  Q: {ratio_Q:.3f}")
        print(f"  U: {ratio_U:.3f}")
        print(f"  V: {ratio_V:.3f}")
        print(f"  Overall: {calibration_ratio:.3f}")

    if calibration_ratio > 1.2:
        print(
            f"  ⚠️  WARNING: Residuals are {calibration_ratio:.1f}x larger than assumed!"
        )
        print("      Model is underfitting or σ_obs is too small.")
    elif calibration_ratio < 0.8:
        print(
            f"  ⚠️  WARNING: Residuals are {calibration_ratio:.1f}x smaller than assumed!"
        )
        print("      Model is overfitting or σ_obs is too large.")
    else:
        print("  ✓ Good calibration (within 20% of assumed value)")

    return {
        "sigma_obs_assumed": sigma_obs_used,
        "std_Q": std_Q,
        "std_U": std_U,
        "std_V": std_V,
        "std_all": std_all,
        "std_per_pixel_mean": std_per_pixel_mean,
        "std_per_pixel_median": std_per_pixel_median,
        "calibration_ratio": calibration_ratio,
    }


# =============================================================================
# 3. Run Comparisons
# =============================================================================

results = {}

# --- WITHOUT REGULARIZATION ---
if RUN_WITHOUT_REGULARIZATION:
    print("\n" + "=" * 60)
    print("RUNNING WITHOUT REGULARIZATION")
    print("=" * 60)

    # Determine sigma mode
    sigma_obs = None if USE_LEARNED_SIGMA else SIGMA_FROM_DETERMINISTIC

    det_noreg = run_deterministic(
        problem, ny, nx, regu_v=0.0, regu_qu=0.0, label="no_reg"
    )
    results["det_noreg"] = det_noreg

    init_vals = [-1.0, -3.0, -5.0, -7.0, -9.0]

    for init_val in init_vals:
        print(f"\n--- SWEEP (No Reg): init_logvar = {init_val} ---")
        prob_noreg = run_probabilistic(
            problem,
            sigma_obs=sigma_obs,
            regu_blos=0.0,
            regu_bqu=0.0,
            label=f"no_reg_{init_val}",
            init_logvar=init_val,
        )

        plot_comparison(
            det_noreg,
            prob_noreg,
            title_suffix=f" (No Reg, Init {init_val})",
            save_name=f"comparison_init_{init_val}_no_reg.png",
        )

        n_stokes = 3  # Q, U, V only
        n_active_wav = (
            len(problem.active_wav_idx)
            if problem.active_wav_idx is not None
            else problem.obs.n_lambda
        )
        n_points = 35600 * n_stokes * n_active_wav
        target_extensive = 0.5 * (1.0 + np.log(0.0121**2)) * n_points

        plot_dual_dynamics(
            prob_noreg["history"],
            target_nll_extensive=target_extensive,
            title_suffix=f" (No Reg, Init {init_val})",
            save_name=f"dynamics_init_{init_val}_no_reg.png",
        )

        save_history_json(
            prob_noreg["history"], save_name=f"history_init_{init_val}_no_reg.json"
        )

        print_metrics(det_noreg, prob_noreg, label=f"(No Reg, Init {init_val})")

        results[f"prob_noreg_{init_val}"] = prob_noreg


# --- WITH REGULARIZATION (SWEEP) ---
if RUN_WITH_REGULARIZATION:
    print("\n" + "=" * 60)
    print("RUNNING WITH REGULARIZATION (SWEEP)")
    print("=" * 60)

    # Determine sigma mode
    sigma_obs = None if USE_LEARNED_SIGMA else SIGMA_FROM_DETERMINISTIC

    # Run Deterministic ONCE
    det_reg = run_deterministic(
        problem, ny, nx, regu_v=REGU_V, regu_qu=REGU_QU, label="reg"
    )

    init_vals = [-1.0, -3.0, -5.0, -7.0, -9.0]

    for init_val in init_vals:
        print(f"\n--- SWEEP: init_logvar = {init_val} ---")
        # 2. Probabilistic with Regularization
        prob_reg = run_probabilistic(
            problem,
            sigma_obs=sigma_obs,
            regu_blos=1.0e5,
            regu_bqu=0.5e5,  # Retuned for extensive scale (NLL ~ 1e7)
            label=f"prob_reg_{init_val}",
            init_logvar=init_val,
        )

        plot_comparison(
            det_reg,
            prob_reg,
            title_suffix=f" (Init {init_val})",
            save_name=f"comparison_init_{init_val}_with_reg.png",
        )

        n_stokes = 3  # Q, U, V only
        n_active_wav = (
            len(problem.active_wav_idx)
            if problem.active_wav_idx is not None
            else problem.obs.n_lambda
        )
        n_points = 35600 * n_stokes * n_active_wav
        target_extensive = 0.5 * (1.0 + np.log(0.0121**2)) * n_points

        plot_dual_dynamics(
            prob_reg["history"],
            target_nll_extensive=target_extensive,
            title_suffix=f" (Init {init_val})",
            save_name=f"dynamics_init_{init_val}.png",
        )

        save_history_json(
            prob_reg["history"], save_name=f"history_init_{init_val}.json"
        )

        print_metrics(det_reg, prob_reg, label=f"(Init {init_val})")

        results[f"prob_reg_{init_val}"] = prob_reg

    results["det_reg"] = det_reg


# =============================================================================
# 4. Summary
# =============================================================================
print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)
print("Output files:")
for val in init_vals:
    print(f"  - comparison_init_{val}_with_reg.png")
    print(f"  - dynamics_init_{val}.png")
    print(f"  - history_init_{val}.json")
print("\nDone!")
