# frf_lasso

A Python library for fitting electrochemical impedance spectra (EIS) to rational polynomial models with L1 (LASSO) regularization, built on top of [lmfit](https://lmfit.github.io/lmfit-py/).

---

## Background

Electrochemical impedance spectroscopy measures a complex-valued impedance Z(ω) across a range of frequencies. A flexible, physics-agnostic way to model this data is a **rational polynomial** (also called a pole-zero model):

```
Z(ω) = P(s) / Q(s)

where:
  s = √(i·ω)
  P(s) = a₀ + a₁s + a₂s² + ... + aₙsⁿ   (numerator)
  Q(s) = 1  + b₁s + b₂s² + ... + bₙsⁿ   (denominator)
```

The model order `n` controls the complexity of the fit and must be chosen by the user (typically by comparing fit statistics across several orders).

**Why LASSO regularization?** Without it, the numerator and denominator polynomials tend to compensate each other, driving coefficients to arbitrarily large values while the ratio remains finite. L1 regularization penalizes the magnitude of all coefficients and keeps the optimization well-conditioned. The strength of the penalty is controlled by `reg_factor` and must also be chosen by the user by comparing statistics.

> **Frequency convention** — all functions in this library expect **angular frequency ω (rad/s)**.
> Instruments typically report frequency in Hz; convert once before passing data to any library function:
> `omega = 2 * np.pi * freq_hz`.

---

## Features

- Rational polynomial models of orders 4 through 9
- L1-regularized least-squares fitting via `lmfit`
- Three fitting strategies:
  - **Single spectrum** — fit one impedance dataset
  - **Multi-start** — repeat with random initializations to assess solution uniqueness
  - **Sequential** — fit a time-series of spectra, using each result to warm-start the next
- Structured save/load for all fit types, with metadata for later retrieval
- Chi-square statistics and consistency metrics for model selection
- Interactive Nyquist plot with spectrum slider

---

## Installation

The library is not yet on PyPI. Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/frf_lasso.git
cd frf_lasso
pip install -e .
```

### Dependencies

```
numpy
lmfit
matplotlib
```

---

## Quick start

The library is **format-agnostic**: it expects data already loaded as NumPy arrays. Loading from your specific file format (`.npy`, `.csv`, `.txt`, etc.) is your responsibility and belongs in your script.

```python
import numpy as np
from frf_lasso import fit_single, fit_multistart, fit_sequential
from frf_lasso.io import save_single, load_single
from frf_lasso.statistics import compare_fits
from frf_lasso.visualization import nyquist_plot

# --- Load your data (format is up to you) ---
freq_hz   = np.load("frequencies.npy")            # shape: (N,), in Hz
omega     = 2 * np.pi * freq_hz                   # convert to rad/s — required by frf_lasso
impedance = np.load("impedance.npy")              # shape: (N,), complex

# --- Fit a single spectrum ---
result, fit = fit_single(
    omega,
    impedance,
    order=6,
    reg_factor=1e-8,
)

# --- Inspect the result (standard lmfit interface) ---
print(result.fit_report())

# --- Save for later ---
save_single("./results/my_fit", frequencies, impedance, result, fit, order=6, reg_factor=1e-8)

# --- Visualize ---
nyquist_plot(frequencies, impedance, fit=fit)
```

---

## Supplying weighting factors

All fitting functions require the caller to supply a weighting array explicitly. The choice of weighting strategy is problem-dependent and left entirely to the user.

| Function | Expected shape |
|---|---|
| `fit_single` | `(N,)` — one weight per frequency |
| `fit_multistart` | `(N,)` — one weight per frequency |
| `fit_sequential` | `(N,)` same weights for all spectra, or `(N, T)` one column per spectrum |
| `fit_simultaneous` | `(N,)` same weights for all spectra |

Common choices include:

```python
# Unit weights — all frequencies equally weighted
weights = np.ones(len(omega))

# Modulus weights — reduces influence of high-impedance points
weights = 1 / np.abs(impedance) ** 0.5

# For sequential fitting with per-spectrum modulus weights
weights = 1 / np.abs(impedance_set) ** 0.5   # shape (N, T)
```

---

## Choosing model order and regularization factor

Neither `order` nor `reg_factor` has a universally correct value. The typical workflow is to sweep over both and compare the fit statistics:

```python
from frf_lasso import fit_single
from frf_lasso.statistics import compare_fits

results = []
for order in [4, 5, 6, 7, 8]:
    for reg_factor in [1e-10, 1e-8, 1e-6]:
        result, fit = fit_single(frequencies, impedance, order=order, reg_factor=reg_factor)
        results.append((order, reg_factor, result, fit))

# Print a summary table (chi2, AIC, BIC, n_params)
compare_fits(results)
```

A good fit has a low chi-square without requiring an unnecessarily high polynomial order (use AIC/BIC to penalize complexity). If chi-square keeps improving as you lower `reg_factor` toward zero and the fit becomes visually poor, the regularization is doing its job.

---

## Fitting strategies

### 1. Single spectrum

Fits one impedance dataset with one set of initial parameters. Uses a global search (basin-hopping) followed by local refinement (Levenberg-Marquardt).

```python
result, fit = fit_single(frequencies, impedance, order=6, reg_factor=1e-8)
```

### 2. Multi-start

Repeats the single-spectrum fit `n_starts` times with different random initializations. Useful for checking whether the solution is unique or whether the objective function has multiple local minima.

```python
results, fits = fit_multistart(
    frequencies, impedance,
    order=6, reg_factor=1e-8,
    n_starts=50,
)

# Check consistency across starts
from frf_lasso.statistics import consistency_metrics
metrics = consistency_metrics(results)
print(metrics)  # CV, ratio of fits within 2× best chi2, etc.
```

### 3. Sequential fit

Fits a time-ordered collection of spectra one by one, using the previous result as the starting point for the next. Suitable for battery cycling experiments or any dataset where the impedance evolves slowly over time.

```python
# impedance_set: shape (N_freq, N_spectra), complex
impedance_set = np.load("impedance_set.npy")

results, fits = fit_sequential(
    frequencies, impedance_set,
    order=6, reg_factor=1e-8,
)
```

---

## Saving and loading results

Each fit session is saved as a folder containing the original data, the model metadata, and the lmfit result. This makes it easy to reload and compare sessions later without re-running the fitting.

```
my_fit_session/
  metadata.json      ← model_order, reg_factor, chi2, AIC, BIC, fit_type, ...
  impedance.npy      ← original complex impedance
  result.json        ← full lmfit result (params, statistics, covariance, ...)
  fit.npy            ← fitted impedance values
```

For multi-start and sequential fits the layout is extended:

```
my_multistart_session/
  metadata.json
  impedance.npy
  fits/
    fit_000.npy
    fit_001.npy
    ...
  results/
    result_000.json
    result_001.json
    ...
```

### Save/load API

| Fit type | Save | Load |
|---|---|---|
| Single | `save_single(path, ...)` | `load_single(path)` |
| Multi-start | `save_multistart(path, ...)` | `load_multistart(path)` |
| Sequential | `save_sequential(path, ...)` | `load_sequential(path)` |
| Simultaneous | `save_simultaneous(path, ...)` | `load_simultaneous(path)` |

```python
from frf_lasso.io import save_sequential, load_sequential

save_sequential("./results/cycle1", frequencies, impedance_set, results, fits,
                order=6, reg_factor=1e-8)

# In a later session:
frequencies, impedance_set, results, fits, metadata = load_sequential("./results/cycle1")
```

---

## Package structure

```
frf_lasso/
  __init__.py           Public API (fit_single, fit_multistart, fit_sequential)
  models.py             rational_poly(freq, order, a_coeffs, b_coeffs) → complex Z
  objective.py          Residual functions (L1 regularization, temporal smoothing)
  transformations.py    Internal log ↔ linear parameter space conversion
  fitting.py            fit_single(), fit_multistart(), fit_sequential()
  simultaneous.py       simultaneous_fit() — global joint optimization (experimental)
  io.py                 save_*/load_* for all four fit types
  statistics.py         chi_square(), consistency_metrics(), compare_fits()
  visualization.py      nyquist_plot(), error_plot(), slider_plot()

scripts/
  fit_single_spectrum.py     Example: single fit, sweep over order and reg_factor
  fit_multistart.py          Example: multi-start with consistency analysis
  fit_sequential.py          Example: sequential fit across a time-series
  fit_simultaneous.py        Example: global simultaneous fit (experimental)
```

---

## Result objects

All fitting functions return standard `lmfit.MinimizerResult` objects. Refer to the [lmfit documentation](https://lmfit.github.io/lmfit-py/fitting.html#minimizerresult-the-optimization-result) for the full interface. Key attributes:

```python
result.params          # fitted parameters (lmfit.Parameters)
result.chisqr          # chi-square
result.aic             # Akaike information criterion
result.bic             # Bayesian information criterion
result.nfev            # number of function evaluations
result.success         # convergence flag
result.fit_report()    # formatted summary string
```

---

## Notes on the simultaneous fit

`simultaneous_fit()` jointly optimizes all spectra in a time-series in a single optimization run, adding temporal smoothness penalties (1st and 2nd derivative of parameter evolution) to the objective. In practice this approach is significantly slower and did not yield better results than the sequential fit on the test datasets. It is included for completeness and experimentation. The output is a single `MinimizerResult` with one parameter set per spectrum.

---

## License

MIT
