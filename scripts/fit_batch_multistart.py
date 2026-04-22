"""
Script — Batch multi-start fit across all discharge spectra.

Fits every spectrum independently from multiple random starting points.
Only the best result per spectrum (lowest chi-square) is kept.  Per-spectrum
consistency metrics (coefficient of variation, consistency ratio) are computed
to flag spectra where the optimisation landscape may have multiple local minima.

This is the most robust fitting strategy but also the most expensive:
runtime scales as n_spectra × N_STARTS.  With the default N_STARTS=10 and
318 spectra the fit completes in a few minutes; increase N_STARTS to 50–100
for a thorough reliability check.

Set ORDER and REG_FACTOR to the values chosen in the model-selection script.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from frf_lasso import fit_batch_multistart, make_lmfit_model
from frf_lasso.statistics import print_batch_multistart_summary
from frf_lasso.visualization import (
    batch_multistart_cv_plot,
    param_evolution_plot,
    slider_plot,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Parameters — update based on model-selection results
# ---------------------------------------------------------------------------
ORDER      = 6
REG_FACTOR = 1e-8
N_STARTS   = 10   # increase to 50–100 for a thorough reliability check
SEED       = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
freq_hz       = np.loadtxt(DATA_DIR / "frequencies.txt", delimiter=",")
omega         = 2 * np.pi * freq_hz
impedance_set = np.load(DATA_DIR / "non-stationary_impedance_discharge.npy")

n_freq, n_spectra = impedance_set.shape
print(f"Dataset: {n_spectra} spectra, {n_freq} frequencies")

# Per-spectrum modulus weighting — shape (N, T), one column per spectrum
weights = 1 / np.abs(impedance_set) ** 0.5

# ---------------------------------------------------------------------------
# Batch multi-start fit
# ---------------------------------------------------------------------------
print(f"Running batch multi-start fit — order={ORDER}, reg={REG_FACTOR:.0e}, "
      f"n_starts={N_STARTS} × {n_spectra} spectra ...")

model = make_lmfit_model(num_order=ORDER, variable="sqrt_jw")

results, fits, consistency = fit_batch_multistart(
    omega, impedance_set, model, weights,
    n_starts=N_STARTS,
    reg_factor=REG_FACTOR,
    seed=SEED,
)

chi2_values = [r.chisqr for r in results]
print(f"Done.  chi2 — mean: {np.mean(chi2_values):.4e}  "
      f"median: {np.median(chi2_values):.4e}  "
      f"max: {np.max(chi2_values):.4e}")

# ---------------------------------------------------------------------------
# Consistency report
# ---------------------------------------------------------------------------
print_batch_multistart_summary(consistency)

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
fig1, _ = batch_multistart_cv_plot(consistency)
fig1.suptitle(f"Per-spectrum reliability — order={ORDER}, reg={REG_FACTOR:.0e}, "
              f"n_starts={N_STARTS}")

fig2, _ = param_evolution_plot({"batch_multistart": results})
fig2.suptitle(f"Parameter evolution — order={ORDER}, reg={REG_FACTOR:.0e}")

print("\nOpening interactive Nyquist viewer (close window to exit)...")
fig3, ax3, slider, button = slider_plot(impedance_set, fits=fits)

plt.show()
