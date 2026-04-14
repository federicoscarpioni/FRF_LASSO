"""
Visualization functions for impedance fitting results.

All plotting functions return the figure and axes objects so callers can
further customise the appearance or save to disk.  Interactive functions
(those with a slider) additionally return the widget objects, which must
be kept alive in the caller's scope to prevent garbage collection.

Functions
---------
Single spectrum
    nyquist_plot          — Nyquist plot with optional fit overlay
    residual_plot         — weighted residuals vs frequency

Multi-start
    multistart_plot       — chi2 distribution across random starts
    print_multistart_summary — formatted console report

Time-series
    slider_plot           — interactive Nyquist with spectrum slider
    param_evolution_plot  — parameter evolution across time, multiple series
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# ---------------------------------------------------------------------------
# Single spectrum
# ---------------------------------------------------------------------------

def nyquist_plot(
    impedance: np.ndarray,
    fit: np.ndarray = None,
    ax: plt.Axes = None,
) -> tuple:
    """
    Nyquist plot of a single impedance spectrum.

    Parameters
    ----------
    impedance : ndarray, shape (N,), complex
        Measured impedance.
    fit : ndarray, shape (N,), complex or None
        Fitted impedance to overlay.  Not plotted when None.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.  A new figure is created when None.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(impedance.real, -impedance.imag, "o", label="Data")
    if fit is not None:
        ax.plot(fit.real, -fit.imag, "-", label="Fit")
        ax.legend()

    ax.set_aspect("equal")
    ax.set_xlabel("$Z_{\\mathrm{real}}$ / Ω")
    ax.set_ylabel("$-Z_{\\mathrm{imag}}$ / Ω")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def residual_plot(
    omega: np.ndarray,
    impedance: np.ndarray,
    fit: np.ndarray,
    weights: np.ndarray = None,
    ax: plt.Axes = None,
) -> tuple:
    """
    Plot weighted residuals (real and imaginary) vs angular frequency.

    Residuals are defined as ``weights * (fit - data)``.  When ``weights``
    is None, unweighted residuals are shown.  Both panels share the same
    x-axis on a log scale, making it easy to spot frequency ranges where
    the fit is poor.

    Parameters
    ----------
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance : ndarray, shape (N,), complex
        Measured impedance.
    fit : ndarray, shape (N,), complex
        Fitted impedance.
    weights : ndarray, shape (N,), real or None
        Weighting factors.  Unweighted residuals are shown when None.
    ax : matplotlib.axes.Axes or None
        If provided, must be a length-2 array of Axes (real, imag).
        A new figure with two vertically stacked panels is created when None.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes (shape (2,))
    """
    w = weights if weights is not None else np.ones(len(omega))

    res_real = w * (fit.real - impedance.real)
    res_imag = w * (fit.imag - impedance.imag)

    if ax is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    else:
        fig, axes = ax.flat[0].figure, ax

    axes[0].semilogx(omega, res_real, "o-", markersize=3)
    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Weighted residual\n(real)")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(omega, res_imag, "o-", markersize=3, color="tab:orange")
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Weighted residual\n(imag)")
    axes[1].set_xlabel("ω / rad s⁻¹")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Multi-start
# ---------------------------------------------------------------------------

def print_multistart_summary(stats: dict) -> None:
    """
    Print a formatted summary of multi-start chi-square statistics.

    Parameters
    ----------
    stats : dict
        Output of ``statistics.multistart_statistics``.
    """
    sep = "=" * 55
    print(sep)
    print("          MULTI-START CHI-SQUARE SUMMARY")
    print(sep)
    print(f"  Starts:               {stats['n_starts']}")
    print(f"  Min chi2:             {stats['min']:.4e}  (trial {stats['best_index'] + 1})")
    print(f"  Max chi2:             {stats['max']:.4e}")
    print(f"  Mean chi2:            {stats['mean']:.4e}")
    print(f"  Std chi2:             {stats['std']:.4e}")
    print(f"  Median chi2:          {stats['median']:.4e}")
    print(f"  IQR:                  {stats['iqr']:.4e}")
    print(f"  CV:                   {stats['cv']:.4f}")
    print(f"  Consistency ratio:    {stats['consistency_ratio']:.2%}  "
          f"(trials within 2x best)")
    print()

    cv = stats["cv"]
    cr = stats["consistency_ratio"]

    if cv < 0.05:
        cv_label = "excellent (CV < 5%)"
    elif cv < 0.10:
        cv_label = "good (CV < 10%)"
    elif cv < 0.20:
        cv_label = "fair (CV < 20%)"
    else:
        cv_label = "poor — possible multiple minima (CV >= 20%)"

    if cr > 0.80:
        cr_label = "high — most trials converge to the same minimum"
    elif cr > 0.60:
        cr_label = "moderate — some scatter in solutions"
    else:
        cr_label = "low — optimisation may be unreliable"

    print(f"  Variability:  {cv_label}")
    print(f"  Consistency:  {cr_label}")
    print(sep)


def multistart_plot(stats: dict) -> tuple:
    """
    Visualize the chi-square distribution across multi-start trials.

    Three panels:
      - Histogram of chi2 values with mean and median markers
      - Box plot
      - chi2 vs trial index with the 2x-best threshold marked

    Parameters
    ----------
    stats : dict
        Output of ``statistics.multistart_statistics``.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes (shape (3,))
    """
    chi2 = stats["chi2_values"]
    use_log = stats["max"] / stats["min"] > 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Histogram
    axes[0].hist(chi2, bins=min(10, len(chi2)), alpha=0.7, edgecolor="k")
    axes[0].axvline(stats["mean"],   color="tab:red",   linestyle="--",
                    label=f"Mean {stats['mean']:.2e}")
    axes[0].axvline(stats["median"], color="tab:green", linestyle="--",
                    label=f"Median {stats['median']:.2e}")
    axes[0].set_xlabel("χ²")
    axes[0].set_ylabel("Count")
    axes[0].set_title("χ² distribution")
    axes[0].legend(fontsize=8)
    if use_log:
        axes[0].set_yscale("log")

    # Box plot
    axes[1].boxplot(chi2, vert=True)
    axes[1].set_ylabel("χ²")
    axes[1].set_title("χ² box plot")
    axes[1].set_xticks([])
    if use_log:
        axes[1].set_yscale("log")

    # Trial sequence
    axes[2].plot(range(1, len(chi2) + 1), chi2, "o-", markersize=4)
    axes[2].axhline(stats["mean"],         color="tab:red",    linestyle="--",
                    alpha=0.6, label="Mean")
    axes[2].axhline(2 * stats["min"],      color="tab:orange", linestyle=":",
                    label="2x best")
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("χ²")
    axes[2].set_title("χ² vs trial index")
    axes[2].legend(fontsize=8)
    if use_log:
        axes[2].set_yscale("log")

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------

def slider_plot(
    impedance_set: np.ndarray,
    fits: list = None,
) -> tuple:
    """
    Interactive Nyquist plot with a slider to step through spectra.

    Parameters
    ----------
    impedance_set : ndarray, shape (N, T), complex
        Measured impedance at T time points.
    fits : list of ndarray, each shape (N,), complex or None
        Fitted impedance values.  Not plotted when None.

    Returns
    -------
    fig, ax, slider, button : Figure, Axes, Slider, Button
        Widget objects must be kept alive in the caller's scope.
    """
    n_spectra = impedance_set.shape[1]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    z0 = impedance_set[:, 0]
    line_data, = ax.plot(z0.real, -z0.imag, "o", label="Data")
    if fits is not None:
        line_fit, = ax.plot(fits[0].real, -fits[0].imag, "-", label="Fit")
        ax.legend()

    ax.set_aspect("equal")
    ax.set_xlabel("$Z_{\\mathrm{real}}$ / Ω")
    ax.set_ylabel("$-Z_{\\mathrm{imag}}$ / Ω")
    ax.grid(True, alpha=0.3)

    ax_slider = plt.axes([0.1, 0.08, 0.65, 0.03])
    slider = Slider(ax_slider, label="Spectrum", valmin=1, valmax=n_spectra,
                    valinit=1, valfmt="%d", valstep=1)

    ax_button = plt.axes([0.82, 0.05, 0.12, 0.06])
    button = Button(ax_button, "Auto lim", hovercolor="tab:blue")

    def _update(val):
        i = int(slider.val) - 1
        z = impedance_set[:, i]
        line_data.set_xdata(z.real)
        line_data.set_ydata(-z.imag)
        if fits is not None:
            line_fit.set_xdata(fits[i].real)
            line_fit.set_ydata(-fits[i].imag)
        fig.canvas.draw_idle()

    def _autolim(_):
        ax.relim(visible_only=True)
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    button.on_clicked(_autolim)

    plt.show()
    return fig, ax, slider, button


def param_evolution_plot(
    series: dict,
    param_names: list = None,
) -> tuple:
    """
    Plot the evolution of fitted parameters across a time-series.

    Accepts multiple result sets so that sequential and simultaneous fits
    can be compared directly on the same axes.

    Parameters
    ----------
    series : dict
        Mapping of ``{label: results_list}`` where each ``results_list`` is
        the per-spectrum results from ``fit_sequential`` or ``fit_simultaneous``.
        Example: ``{"sequential": seq_results, "simultaneous": sim_results}``.
    param_names : list of str or None
        Parameters to plot.  Defaults to all parameters in the first result
        set when None.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
    """
    first_results = next(iter(series.values()))
    if param_names is None:
        param_names = list(first_results[0].params.keys())

    n_params = len(param_names)
    n_cols = min(5, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 2.5 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes_flat[i]
        for label, results in series.items():
            values = [r.params[name].value for r in results]
            ax.plot(values, marker="o", markersize=3, linewidth=1, label=label)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Spectrum index", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(n_params, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return fig, axes
