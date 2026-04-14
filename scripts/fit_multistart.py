"""
Script 3 — Multi-start fit: solution uniqueness check.

Fits the same spectrum from many random starting points.  A low coefficient
of variation (CV) and high consistency ratio mean the optimisation landscape
has a well-defined global minimum and the chosen ORDER / REG_FACTOR are
reliable.  Large CV or low consistency indicate multiple local minima —
try increasing REG_FACTOR or reducing ORDER.

Set ORDER and REG_FACTOR to the values chosen in script 2.
"""

from pathlib import Path

import lmfit
import numpy as np
import matplotlib.pyplot as plt

from frf_lasso import fit_multistart, make_lmfit_model
from frf_lasso.statistics import multistart_statistics
from frf_lasso.visualization import multistart_plot, print_multistart_summary, nyquist_plot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Parameters — update based on script 2 results
# ---------------------------------------------------------------------------
ORDER      = 6
REG_FACTOR = 1e-8
N_STARTS   = 20
SEED       = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
freq_hz       = np.loadtxt(DATA_DIR / "frequencies.txt", delimiter=",")
omega         = 2 * np.pi * freq_hz
impedance_set = np.load(DATA_DIR / "non-stationary_impedance_discharge.npy")

spectrum_index = impedance_set.shape[1] // 2
impedance      = impedance_set[:, spectrum_index]
weights        = 1 / np.abs(impedance) ** 0.5

print(f"Multi-start fit — spectrum {spectrum_index}, "
      f"order={ORDER}, reg={REG_FACTOR:.0e}, n_starts={N_STARTS}")

# ---------------------------------------------------------------------------
# Multi-start fit
# ---------------------------------------------------------------------------
model = make_lmfit_model(num_order=ORDER)

results, fits = fit_multistart(
    omega, impedance, model, weights,
    n_starts=N_STARTS,
    reg_factor=REG_FACTOR,
    seed=SEED,
)

# ---------------------------------------------------------------------------
# Consistency report
# ---------------------------------------------------------------------------
stats = multistart_statistics(results)
print_multistart_summary(stats)

best_result = results[stats["best_index"]]
best_fit    = fits[stats["best_index"]]
print(f"\nBest result (trial {stats['best_index'] + 1}):")
print(lmfit.fit_report(best_result))

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
fig1, _ = multistart_plot(stats)
fig1.suptitle(f"Chi-square distribution — {N_STARTS} starts, "
              f"order={ORDER}, reg={REG_FACTOR:.0e}")

fig2, ax2 = nyquist_plot(impedance, fit=best_fit)
ax2.set_title(f"Best fit (trial {stats['best_index'] + 1}, "
              f"chi2={best_result.chisqr:.4e})")

plt.show()
