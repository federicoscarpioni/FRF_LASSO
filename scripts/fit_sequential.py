"""
Script 4 — Sequential fit across all discharge spectra.

Fits all spectra in time order, using each result as the warm start for
the next.  The middle spectrum is fitted first to get a better initial
point than a flat guess.

Set ORDER and REG_FACTOR to the values chosen in script 2.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from frf_lasso import fit_single, fit_sequential, make_lmfit_model
from frf_lasso.statistics import smoothness_metrics
from frf_lasso.visualization import slider_plot, param_evolution_plot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Parameters — update based on script 2 results
# ---------------------------------------------------------------------------
ORDER      = 6
REG_FACTOR = 1e-8

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
# Initialise from the middle spectrum for a better warm start
# ---------------------------------------------------------------------------
print(f"Fitting with order={ORDER}, reg={REG_FACTOR:.0e}")

model  = make_lmfit_model(num_order=ORDER)
params = model.make_params()
for name in model.param_names:
    params[name].set(value=1.0, min=1e-9, max=1e6)

mid = n_spectra // 2
init_result, _ = fit_single(omega, impedance_set[:, mid], model, params,
                             weights[:, mid], reg_factor=REG_FACTOR)
init_params = init_result.params.copy()
print(f"Initialisation fit (spectrum {mid}): chi2={init_result.chisqr:.4e}")

# ---------------------------------------------------------------------------
# Sequential fit
# ---------------------------------------------------------------------------
print("Running sequential fit...")
results, fits = fit_sequential(omega, impedance_set, model, init_params,
                               weights, reg_factor=REG_FACTOR)

chi2_values = [r.chisqr for r in results]
print(f"Done.  chi2 — mean: {np.mean(chi2_values):.4e}  "
      f"median: {np.median(chi2_values):.4e}  "
      f"max: {np.max(chi2_values):.4e}")

sm = smoothness_metrics(results)
print("\nParameter smoothness (curvature — lower is smoother):")
for name, m in sm.items():
    print(f"  {name}: {m['curvature']:.4e}")

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
fig1, _ = param_evolution_plot({"sequential": results})
fig1.suptitle(f"Parameter evolution — order={ORDER}, reg={REG_FACTOR:.0e}")

print("\nOpening interactive Nyquist viewer (close window to exit)...")
fig2, ax2, slider, button = slider_plot(impedance_set, fits=fits)

plt.show()
