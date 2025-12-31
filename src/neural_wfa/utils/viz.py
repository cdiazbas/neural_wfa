import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Legacy Plot Parameters (for visual parity)
LEGACY_PARAMS = {
    "_internal.classic_mode": False,
    "agg.path.chunksize": 0,
    "axes.autolimit_mode": "data",
    "axes.axisbelow": "line",
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "axes.formatter.limits": [-7, 7],
    "axes.formatter.min_exponent": 0,
    "axes.formatter.offset_threshold": 4,
    "axes.formatter.use_locale": False,
    "axes.formatter.use_mathtext": False,
    "axes.formatter.useoffset": True,
    "axes.grid": False,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.labelcolor": "black",
    "axes.labelpad": 10.0,
    "axes.labelsize": 19.0,
    "axes.labelweight": "normal",
    "axes.linewidth": 1.8,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.titlepad": 7.5,
    "axes.titlesize": 18.0,
    "axes.titleweight": "normal",
    "axes.unicode_minus": True,
    "axes.xmargin": 0.05,
    "axes.ymargin": 0.05,
    "axes3d.grid": True,
    "backend_fallback": True,
    "boxplot.bootstrap": None,
    "boxplot.boxprops.color": "black",
    "boxplot.boxprops.linestyle": "-",
    "boxplot.boxprops.linewidth": 1.0,
    "boxplot.capprops.color": "black",
    "boxplot.capprops.linestyle": "-",
    "boxplot.capprops.linewidth": 1.0,
    "boxplot.flierprops.color": "black",
    "boxplot.flierprops.linestyle": "none",
    "boxplot.flierprops.linewidth": 1.0,
    "boxplot.flierprops.marker": "o",
    "boxplot.flierprops.markeredgecolor": "black",
    "boxplot.flierprops.markerfacecolor": "none",
    "boxplot.flierprops.markersize": 6.0,
    "boxplot.meanline": False,
    "boxplot.meanprops.color": "C2",
    "boxplot.meanprops.linestyle": "--",
    "boxplot.meanprops.linewidth": 1.0,
    "boxplot.meanprops.marker": "^",
    "boxplot.meanprops.markeredgecolor": "C2",
    "boxplot.meanprops.markerfacecolor": "C2",
    "boxplot.meanprops.markersize": 6.0,
    "boxplot.medianprops.color": "C1",
    "boxplot.medianprops.linestyle": "-",
    "boxplot.medianprops.linewidth": 1.0,
    "boxplot.notch": False,
    "boxplot.patchartist": False,
    "boxplot.showbox": True,
    "boxplot.showcaps": True,
    "boxplot.showfliers": True,
    "boxplot.showmeans": False,
    "boxplot.vertical": True,
    "boxplot.whiskerprops.color": "black",
    "boxplot.whiskerprops.linestyle": "-",
    "boxplot.whiskerprops.linewidth": 1.0,
    "boxplot.whiskers": 1.5,
    "contour.corner_mask": True,
    "contour.negative_linestyle": "dashed",
    "date.autoformatter.day": "%Y-%m-%d",
    "date.autoformatter.hour": "%m-%d %H",
    "date.autoformatter.microsecond": "%M:%S.%f",
    "date.autoformatter.minute": "%d %H:%M",
    "date.autoformatter.month": "%Y-%m",
    "date.autoformatter.second": "%H:%M:%S",
    "date.autoformatter.year": "%Y",
    "docstring.hardcopy": False,
    "errorbar.capsize": 0.0,
    "figure.autolayout": False,
    "figure.constrained_layout.h_pad": 0.04167,
    "figure.constrained_layout.hspace": 0.02,
    "figure.constrained_layout.use": False,
    "figure.constrained_layout.w_pad": 0.04167,
    "figure.constrained_layout.wspace": 0.02,
    "figure.edgecolor": (1, 1, 1, 0),
    "figure.facecolor": (1, 1, 1, 0),
    "figure.figsize": [6.153846153846153, 5.0],
    "figure.frameon": True,
    "figure.max_open_warning": 20,
    "figure.subplot.bottom": 0.125,
    "figure.subplot.hspace": 0.2,
    "figure.subplot.left": 0.125,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.88,
    "figure.subplot.wspace": 0.2,
    "figure.titlesize": "large",
    "figure.titleweight": "normal",
    "font.cursive": [
        "Apple Chancery",
        "Textile",
        "Zapf Chancery",
        "Sand",
        "Script MT",
        "Felipa",
        "cursive",
    ],
    "font.family": "DejaVu Sans",
    "font.fantasy": [
        "Comic Sans MS",
        "Chicago",
        "Charcoal",
        "Impact",
        "Western",
        "Humor Sans",
        "xkcd",
        "fantasy",
    ],
    "font.monospace": [
        "DejaVu Sans Mono",
        "Bitstream Vera Sans Mono",
        "Computer Modern Typewriter",
        "Andale Mono",
        "Nimbus Mono L",
        "Courier New",
        "Courier",
        "Fixed",
        "Terminal",
        "monospace",
    ],
    "font.sans-serif": ["CMU Sans Serif"],
    "font.serif": ["CMU Serif"],
    "font.size": 10.0,
    "font.stretch": "normal",
    "font.style": "normal",
    "font.variant": "normal",
    "font.weight": "normal",
    "grid.alpha": 1.0,
    "grid.color": "#b0b0b0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "hatch.color": "black",
    "hatch.linewidth": 1.0,
    "hist.bins": 10,
    "image.aspect": "equal",
    "image.cmap": "viridis",
    "image.composite_image": True,
    "image.interpolation": "nearest",
    "image.lut": 256,
    "image.origin": "upper",
    "image.resample": True,
    "interactive": True,
    "legend.borderaxespad": 0.5,
    "legend.borderpad": 0.4,
    "legend.columnspacing": 2.0,
    "legend.edgecolor": "0.8",
    "legend.facecolor": "inherit",
    "legend.fancybox": True,
    "legend.fontsize": 17.0,
    "legend.framealpha": 0.4,
    "legend.frameon": False,
    "legend.handleheight": 0.7,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.8,
    "legend.labelspacing": 0.5,
    "legend.loc": "best",
    "legend.markerscale": 1.0,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
    "legend.shadow": False,
    "legend.title_fontsize": None,
    "lines.antialiased": True,
    "lines.color": "C0",
    "lines.dash_capstyle": "butt",
    "lines.dash_joinstyle": "round",
    "lines.dashdot_pattern": [6.4, 1.6, 1.0, 1.6],
    "lines.dashed_pattern": [3.7, 1.6],
    "lines.dotted_pattern": [1.0, 1.65],
    "lines.linestyle": "-",
    "lines.linewidth": 2.5,
    "lines.marker": "None",
    "lines.markeredgecolor": "auto",
    "lines.markeredgewidth": 1.0,
    "lines.markerfacecolor": "auto",
    "lines.markersize": 6.0,
    "lines.scale_dashes": True,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "markers.fillstyle": "full",
    "mathtext.bf": "sans:bold",
    "mathtext.cal": "cursive",
    "mathtext.default": "it",
    "mathtext.fontset": "dejavusans",
    "mathtext.it": "sans:italic",
    "mathtext.rm": "sans",
    "mathtext.sf": "sans",
    "mathtext.tt": "monospace",
    "patch.antialiased": True,
    "patch.edgecolor": "black",
    "patch.facecolor": "C0",
    "patch.force_edgecolor": False,
    "patch.linewidth": 1.0,
    "path.effects": [],
    "path.simplify": True,
    "path.simplify_threshold": 0.1111111111111111,
    "path.sketch": None,
    "path.snap": True,
    "pdf.compression": 6,
    "pdf.fonttype": 3,
    "pdf.inheritcolor": False,
    "pdf.use14corefonts": False,
    "pgf.rcfonts": True,
    "pgf.texsystem": "xelatex",
    "polaraxes.grid": True,
    "ps.distiller.res": 6000,
    "ps.fonttype": 3,
    "ps.papersize": "letter",
    "ps.useafm": False,
    "ps.usedistiller": False,
    "savefig.bbox": None,
    "savefig.directory": "~",
    "savefig.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.format": "png",
    "savefig.orientation": "portrait",
    "savefig.pad_inches": 0.1,
    "savefig.transparent": False,
    "scatter.marker": "o",
    "svg.fonttype": "path",
    "svg.hashsalt": None,
    "svg.image_inline": True,
    "text.antialiased": True,
    "text.color": "black",
    "text.hinting": "auto",
    "text.hinting_factor": 8,
    "text.usetex": True,
    "timezone": "UTC",
    "tk.window_focus": False,
    "toolbar": "toolbar2",
    "webagg.address": "127.0.0.1",
    "webagg.open_in_browser": True,
    "webagg.port": 8988,
    "webagg.port_retries": 50,
    "xtick.alignment": "center",
    "xtick.bottom": True,
    "xtick.color": "black",
    "xtick.direction": "out",
    "xtick.labelbottom": True,
    "xtick.labelsize": 17.5,
    "xtick.labeltop": False,
    "xtick.major.bottom": True,
    "xtick.major.pad": 6.0,
    "xtick.major.size": 14.0,
    "xtick.major.top": True,
    "xtick.major.width": 1.0,
    "xtick.minor.bottom": True,
    "xtick.minor.pad": 3.4,
    "xtick.minor.size": 6.0,
    "xtick.minor.top": True,
    "xtick.minor.visible": True,
    "xtick.minor.width": 0.8,
    "xtick.top": False,
    "ytick.alignment": "center_baseline",
    "ytick.color": "black",
    "ytick.direction": "out",
    "ytick.labelleft": True,
    "ytick.labelright": False,
    "ytick.labelsize": 17.5,
    "ytick.major.left": True,
    "ytick.major.pad": 6.0,
    "ytick.major.right": True,
    "ytick.major.size": 14.0,
    "ytick.major.width": 1.0,
    "ytick.minor.left": True,
    "ytick.minor.pad": 3.4,
    "ytick.minor.right": True,
    "ytick.minor.size": 6.0,
    "ytick.minor.visible": True,
    "ytick.minor.width": 0.8,
    "ytick.left": True,
    "ytick.right": False,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{newcent} \boldmath",
}

def set_plot_params(params: dict = None):
    """
    Sets default matplotlib parameters.
    Default (params=None) uses the robust legacy parameters for consistency.
    """
    if params is None:
        params = LEGACY_PARAMS.copy()
        
    # Safety check for text.usetex if latex is not available
    # We simply try to apply, if it fails, we fall back?
    # For now, blindly apply as in legacy code.
    
    # plt.show() # Legacy had this, but it might popup a window. In scripts it's fine.
    # In headless env (Agg), plt.show() does nothing usually.
    
    plt.rcParams.copy() # Legacy did this, implies resets? No, copy() returns dict.
    # Legacy: plt.rcParams.copy(); pylab.rcParams.update(params)
    # The copy() call is useless unless assigned. It might be a remnant.
    # pylab.rcParams update works on global state.
    
    pylab.rcParams.update(params)

# Alias for legacy compatibility
set_params = set_plot_params

def formatting(value):
    """
    Formats a numerical value into a LaTeX-style scientific notation string.
    """
    formatted_str = "{:.0e}".format(value)
    formatted_str = formatted_str.replace("e-0", " \\times 10^{-")
    formatted_str += "}"
    if value == 0:
        formatted_str = "0"
    return formatted_str

def add_colorbar(im, aspect=20, pad_fraction=0.5, nbins=5, orientation='vertical', **kwargs):
    """
    Add a color bar to an image plot.
    Exact legacy implementation.
    """
    from mpl_toolkits import axes_grid1
    from matplotlib import ticker

    divider = axes_grid1.make_axes_locatable(im.axes)
    
    if orientation.lower() == 'horizontal':
        width = axes_grid1.axes_size.AxesX(im.axes, aspect=1.0/aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("bottom", size=width, pad=pad)
        plt.sca(current_ax)
    else:  # vertical (default)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
    
    cb = im.axes.figure.colorbar(im, cax=cax, orientation=orientation, **kwargs)
    tick_locator = ticker.MaxNLocator(nbins)
    cb.locator = tick_locator
    cb.update_ticks()
    return cb

def torch2plot(tensor: torch.Tensor, shape=None) -> np.ndarray:
    """
    Converts a torch tensor to a numpy array and optionally reshapes it for plotting.
    """
    arr = tensor.detach().cpu().numpy()
    if shape and len(arr.shape) == 1:
        return arr.reshape(shape)
    return arr

def plot_loss(output_dict):
    """
    Plot the loss and learning rate during training.
    Strict legacy parity.
    """
    # Loss plot:
    plt.figure()
    
    loss_data = output_dict["loss"]
    if isinstance(loss_data, list):
        loss_data = np.array(loss_data)
        
    plt.plot(loss_data, alpha=0.5)
    
    # Smoothing windows of the 10% of the total number of iterations:
    savgol_loss = loss_data
    if len(loss_data) > 10:
        window = int(len(loss_data) / 10)
        from scipy.signal import savgol_filter
        savgol_loss = savgol_filter(loss_data, window, 3 if window > 3 else 1)
        plt.plot(savgol_loss, "C0-", alpha=0.8)
        plt.plot(savgol_loss, "k-", alpha=0.2)

    if len(loss_data) > 1:
        # Use formatting helper
        output_title_latex = formatting(loss_data[-1])
        # Or strict copy from legacy which reimplemented logic inside
        # Legacy code:
        # output_title_latex = r"${:.2e}".format(output_dict["loss"][-1]).replace("e", "\\times 10^{") + "}$"
        # Since I added formatting(), I can use it, but legacy plot_loss implementation actually didn't use formatting() func call, 
        # it inlined it. "formatting()" func handles "e-0" vs "e". 
        # Legacy plot_loss handles "e".
        # Let's stick to strict legacy copy for plot_loss to be safe.
        output_title_latex = (
            r"${:.2e}".format(loss_data[-1]).replace("e", "\\times 10^{")
            + "}$"
        )
        plt.title("Final loss: " + output_title_latex)
        
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.minorticks_on()
    plt.yscale("log")

    # Another axis with the lr:
    if "lr" in output_dict and output_dict["lr"] is not None and len(output_dict["lr"]) > 0:
        ax2 = plt.gca().twinx()
        ax2.plot(output_dict["lr"], "k--", alpha=0.5)
        ax2.set_yscale("log")
        ax2.set_ylabel("Learning rate")

    return plt.gcf()

# Alias for compatibility if needed (renaming plot_training_progress to plot_loss)
plot_training_progress = plot_loss

def plot_wfa_results(blos, btrans, phi, save_name=None, show=True):
    """
    Plots the Blos, Btrans, and Azimuth maps.
    """
    ny, nx = blos.shape
    fs = (9*1.5, 4.5*1.5)
    extent = np.float64((0, nx, 0, ny))
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=fs)

    im0 = ax[0].imshow(blos, vmax=800, vmin=-800, cmap='RdGy', interpolation='nearest', extent=extent)
    im1 = ax[1].imshow(btrans, vmin=0, vmax=800, cmap='gist_gray', interpolation='nearest', extent=extent)
    im2 = ax[2].imshow(phi, vmax=np.pi, vmin=0, cmap='twilight', interpolation='nearest', extent=extent)

    names = [r'B$_\parallel$', r'B$_\bot$', r'$\Phi_B$']
    add_colorbar(im0, orientation='horizontal', label=names[0]+' [G]', pad_fraction=0.17)
    add_colorbar(im1, orientation='horizontal', label=names[1]+' [G]', pad_fraction=0.17)
    add_colorbar(im2, orientation='horizontal', label=names[2]+' [rad]', pad_fraction=0.17)

    for ii in range(3):
        ax[ii].set_xlabel('x [pixels]')
    ax[0].set_ylabel('y [pixels]')
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)
    if show:
        plt.show()

def plot_stokes_profiles(wav, obs_quv, mod_quv, mask_indices=None, save_name=None, show=True):
    """
    Plots Stokes Q, U, V profiles (Observed vs Model).
    obs_quv: (obs_Q, obs_U, obs_V) tuple of 1D arrays
    mod_quv: (mod_Q, mod_U, mod_V) tuple of 1D arrays
    """
    obs_Q, obs_U, obs_V = obs_quv
    mod_Q, mod_U, mod_V = mod_quv
    
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10*1.5, 4))
    ax = ax.flatten()

    # Q
    ax[0].plot(wav, mod_Q, label='Stokes Q', color='C0')
    ax[0].plot(wav, obs_Q, 'k.', label='Stokes Q')
    ax[0].set_ylabel('Stokes Q [a.u.]')

    # U
    ax[1].plot(wav, mod_U, label='Stokes U', color='C1')
    ax[1].plot(wav, obs_U, 'k.', label='Stokes U')
    ax[1].set_ylabel('Stokes U [a.u.]')

    # V
    ax[2].plot(wav, mod_V, label='Stokes V', color='C2')
    ax[2].plot(wav, obs_V, 'k.', label='Stokes V')
    ax[2].set_xlabel(r"Wavelength")
    ax[2].set_ylabel('Stokes V [a.u.]')

    if mask_indices is not None:
        w_min = wav[mask_indices[0]]
        w_max = wav[mask_indices[-1]]
        for ii in range(3):
            ylim = ax[ii].get_ylim()
            ax[ii].fill_betweenx(
                ylim, w_min, w_max,
                color='C4', alpha=0.2,
                label='Mask used for the estimation',
            )
            
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)
    if show:
        plt.show()

def plot_chi2_maps(chi2_q, chi2_u, chi2_v, chi2_total=None, save_name_components=None, save_name_total=None, show=True):
    """
    Plots the Chi2 spatial maps (Components and Total).
    """
    ny, nx = chi2_q.shape
    fs = (9*1.5, 7)
    extent = np.float64((0, nx, 0, ny))
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=fs, sharex=True, sharey=True)
    ax = ax.flatten()

    im0 = ax[0].imshow(chi2_q, cmap='magma', interpolation='nearest', extent=extent)
    im1 = ax[1].imshow(chi2_u, cmap='magma', interpolation='nearest', extent=extent)
    im2 = ax[2].imshow(chi2_v, cmap='magma', interpolation='nearest', extent=extent)

    names = [r'$\chi^2_Q$', r'$\chi^2_U$', r'$\chi^2_V$']
    # Use helper-like logic or direct calls
    add_colorbar(im0, pad_fraction=0.1, label=names[0]+' [a.u.]', orientation='horizontal')
    add_colorbar(im1, pad_fraction=0.1, label=names[1]+' [a.u.]', orientation='horizontal')
    add_colorbar(im2, pad_fraction=0.1, label=names[2]+' [a.u.]', orientation='horizontal')

    for ii in range(3):
        ax[ii].set_xlabel('x [pixels]')
    ax[0].set_ylabel('y [pixels]')
    plt.tight_layout()
    
    if save_name_components:
        plt.savefig(save_name_components, dpi=300)
    if show:
        plt.show()

    if chi2_total is not None:
        plt.figure(figsize=(6*1.5, 4*1.5))
        im = plt.imshow(chi2_total, cmap='magma', interpolation='nearest', extent=extent)
        plt.colorbar(im, label=r"$\chi^2$ [a.u.]")
        plt.ylabel('y [pixels]')
        plt.xlabel('x [pixels]')
        if save_name_total:
            plt.savefig(save_name_total, dpi=300)
        if show:
            plt.show()

def plot_uncertainties(unc_blos, unc_btrans, unc_phi, save_name=None, show=True):
    """
    Plots uncertainty maps.
    """
    ny, nx = unc_blos.shape
    fs = (9*1.5, 4.5*1.5)
    extent = np.float64((0, nx, 0, ny))
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=fs)

    im0 = ax[0].imshow(unc_blos, extent=extent, aspect='auto', cmap='gray', vmax=1e2)
    im1 = ax[1].imshow(unc_btrans, extent=extent, aspect='auto', cmap='gray', vmax=1e3)
    im2 = ax[2].imshow(unc_phi, extent=extent, aspect='auto', cmap='gray', vmax=3)

    names = [r'$\Delta$ B$_\parallel$', r'$\Delta$ B$_\bot$', r'$\Delta \Phi_B$']
    add_colorbar(im0, orientation='horizontal', label=names[0]+' [G]', pad_fraction=0.17)
    add_colorbar(im1, orientation='horizontal', label=names[1]+' [G]', pad_fraction=0.17)
    add_colorbar(im2, orientation='horizontal', label=names[2]+' [rad]', pad_fraction=0.17)

    for ii in range(3):
        ax[ii].set_xlabel('x [pixels]')
    ax[0].set_ylabel('y [pixels]')
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300)
    if show:
        plt.show()
