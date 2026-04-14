"""
Script 1 — Fit one impedance spectrum.

The simplest possible use of frf_lasso: load one spectrum, fit it with a
chosen model order and regularization factor, and plot the result.

Run this first to check that the package is working and to get a feel for
the API before moving on to the model-selection and sequential scripts.
"""

from pathlib import Path

import lmfit
import numpy as np
import matplotlib.pyplot as plt

from frf_lasso import fit_single, make_lmfit_model
from frf_lasso.visualization import nyquist_plot, residual_plot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
freq_hz       = np.loadtxt(DATA_DIR / "frequencies.txt", delimiter=",")
omega         = 2 * np.pi * freq_hz
impedance_set = np.load(DATA_DIR / "non-stationary_impedance_discharge.npy")

# Pick one spectrum to fit (middle of the discharge sequence)
spectrum_index = impedance_set.shape[1] // 2
impedance      = impedance_set[:, spectrum_index]
weights        = 1 / np.abs(impedance) ** 0.5   # square root modulus weighting

print(f"Fitting spectrum {spectrum_index} "
      f"({freq_hz[0]:.3g} – {freq_hz[-1]:.3g} Hz, {len(freq_hz)} points)")

# ---------------------------------------------------------------------------
# Build the model and set initial parameter values
# ---------------------------------------------------------------------------
ORDER      = 6
REG_FACTOR = 1e-8

model  = make_lmfit_model(num_order=ORDER)
params = model.make_params()
for name in model.param_names:
    params[name].set(value=1.0, min=1e-9, max=1e6)

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
result, fit = fit_single(omega, impedance, model, params, weights,
                         reg_factor=REG_FACTOR)

print(f"\nchi2  = {result.chisqr:.4e}")
print(f"AIC   = {result.aic:.4e}")
print(lmfit.fit_report(result))

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
fig1, ax1 = nyquist_plot(impedance, fit=fit)
ax1.set_title(f"Spectrum {spectrum_index} — order {ORDER}, reg={REG_FACTOR:.0e}")

fig2, _ = residual_plot(omega, impedance, fit)
fig2.suptitle("Percentage residuals")

plt.show()
