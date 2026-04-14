"""
Script 2 — Model order and regularization selection.

Sweeps over a grid of polynomial orders and regularization factors on one
representative spectrum and prints a comparison table sorted by AIC.

Inspect the table, pick the combination that gives the lowest AIC without
overfitting (watch for near-identical AIC values at higher orders — that
is a sign of over-parameterisation), then set ORDER and REG_FACTOR in
fit_single_spectrum.py accordingly.
"""

from pathlib import Path

import numpy as np

from frf_lasso import fit_single, make_lmfit_model
from frf_lasso.statistics import compare_fits

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

spectrum_index = impedance_set.shape[1] // 2
impedance      = impedance_set[:, spectrum_index]
weights        = 1 / np.abs(impedance) ** 0.5

print(f"Dataset: {impedance_set.shape[1]} spectra, {impedance_set.shape[0]} frequencies")
print(f"Fitting spectrum {spectrum_index}\n")

# ---------------------------------------------------------------------------
# Sweep over model orders and regularization factors
# ---------------------------------------------------------------------------
ORDERS      = [4, 5, 6, 7]
REG_FACTORS = [1e-10, 1e-8, 1e-6, 1e-4]

candidates = []

print("Fitting grid...")
for order in ORDERS:
    for reg_factor in REG_FACTORS:
        model  = make_lmfit_model(num_order=order)
        params = model.make_params()
        for name in model.param_names:
            params[name].set(value=1.0, min=1e-9, max=1e6)

        result, _ = fit_single(omega, impedance, model, params, weights,
                               reg_factor=reg_factor)
        candidates.append((order, reg_factor, result))
        print(f"  order={order}  reg={reg_factor:.0e}  "
              f"chi2={result.chisqr:.4e}  AIC={result.aic:.4e}")

# ---------------------------------------------------------------------------
# Comparison table — inspect this to choose ORDER and REG_FACTOR
# ---------------------------------------------------------------------------
print("\n--- Model comparison (sorted by AIC) ---")
compare_fits(candidates)
print("\nSet ORDER and REG_FACTOR in fit_single_spectrum.py based on the table above.")
