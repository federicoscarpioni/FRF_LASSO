"""
Script 5 — Simultaneous global fit: temporal smoothness refinement.

Re-runs the sequential fit for initialisation, then refines all spectra
jointly in a single minimisation with temporal smoothness penalties that
discourage rapid changes in parameter values across time.

Note: this fit is significantly slower than the sequential fit and does
not always produce better results.  The per-spectrum statistics (chisqr,
AIC, BIC) reflect the global joint optimisation, not individual spectra.

Set ORDER, REG_FACTOR, and SMT_FACTOR to the values chosen in scripts 2–4.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from frf_lasso import fit_single, fit_sequential, fit_simultaneous, make_lmfit_model
from frf_lasso.statistics import smoothness_metrics
from frf_lasso.visualization import slider_plot, param_evolution_plot

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Parameters — update based on scripts 2 and 4 results
# ---------------------------------------------------------------------------
ORDER      = 6
REG_FACTOR = 1e-8
SMT_FACTOR = 1e-3

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
freq_hz       = np.loadtxt(DATA_DIR / "frequencies.txt", delimiter=",")
omega         = 2 * np.pi * freq_hz
impedance_set = np.load(DATA_DIR / "non-stationary_impedance_discharge.npy")

n_freq, n_spectra = impedance_set.shape
print(f"Dataset: {n_spectra} spectra, {n_freq} frequencies")

# ---------------------------------------------------------------------------
# Sequential fit (initialisation for the simultaneous fit)
# ---------------------------------------------------------------------------
print(f"\nRunning sequential fit for initialisation (order={ORDER}, reg={REG_FACTOR:.0e})...")

model  = make_lmfit_model(num_order=ORDER)
params = model.make_params()
for name in model.param_names:
    params[name].set(value=1.0, min=1e-9, max=1e6)

seq_weights = 1 / np.abs(impedance_set) ** 0.5   # shape (N, T)

mid = n_spectra // 2
init_result, _ = fit_single(omega, impedance_set[:, mid], model, params,
                             seq_weights[:, mid], reg_factor=REG_FACTOR)
init_params = init_result.params.copy()

seq_results, seq_fits = fit_sequential(omega, impedance_set, model, init_params,
                                       seq_weights, reg_factor=REG_FACTOR)
seq_chi2 = [r.chisqr for r in seq_results]
print(f"Done.  mean chi2: {np.mean(seq_chi2):.4e}")

# ---------------------------------------------------------------------------
# Simultaneous fit
# ---------------------------------------------------------------------------
# Requires a single 1D weight vector — use the mean modulus weighting.
weights_1d = np.mean(1 / np.abs(impedance_set) ** 0.5, axis=1)

print(f"\nRunning simultaneous fit (smt={SMT_FACTOR:.0e}) — this may take several minutes...")

global_result, sim_results, sim_fits = fit_simultaneous(
    omega, impedance_set, model, seq_results, weights_1d,
    reg_factor=REG_FACTOR,
    smt_factor=SMT_FACTOR,
)

print(f"Done.  Global chi2: {global_result.chisqr:.4e}  "
      f"AIC: {global_result.aic:.4e}  BIC: {global_result.bic:.4e}")
print("(statistics reflect the full joint optimisation, not individual spectra)")

# ---------------------------------------------------------------------------
# Smoothness comparison
# ---------------------------------------------------------------------------
sm_seq = smoothness_metrics(seq_results)
sm_sim = smoothness_metrics(sim_results)

print("\nParameter smoothness (curvature — lower is smoother):")
print(f"  {'Parameter':<12}  {'Sequential':>14}  {'Simultaneous':>14}  {'Improvement':>12}")
print(f"  {'-'*56}")
for name in sm_seq:
    c_seq = sm_seq[name]["curvature"]
    c_sim = sm_sim[name]["curvature"]
    ratio = c_seq / c_sim if c_sim > 0 else float("inf")
    print(f"  {name:<12}  {c_seq:>14.4e}  {c_sim:>14.4e}  {ratio:>11.2f}x")

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
fig1, _ = param_evolution_plot(
    {"sequential": seq_results, "simultaneous": sim_results}
)
fig1.suptitle(f"Parameter evolution — order={ORDER}, "
              f"reg={REG_FACTOR:.0e}, smt={SMT_FACTOR:.0e}")

print("\nOpening interactive Nyquist viewer (close window to exit)...")
fig2, ax2, slider, button = slider_plot(impedance_set, fits=sim_fits)
ax2.set_title("Simultaneous fit")

plt.show()
