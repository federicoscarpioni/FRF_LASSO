# frf_lasso

A Python library for fitting electrochemical impedance spectra (EIS) to rational polynomial models with L1 (LASSO) regularization, built on top of [lmfit](https://lmfit.github.io/lmfit-py/).

---

## Background

Electrochemical impedance spectroscopy measures a complex-valued impedance Z(ω) across a range of frequencies. A flexible, physics-agnostic way to model this data is a **rational polynomial** (also called a pole-zero model):

```
Z(ω) = P(s) / Q(s)

where:
  s = √(i·ω)
  P(s) = a₀ + a₁s + a₂s² + ... + aₙsⁿ   (numerator, order n)
  Q(s) = 1  + b₁s + b₂s² + ... + bₘsᵐ   (denominator, order m)
```

The leading denominator coefficient is fixed to 1 to make the model identifiable — without this constraint the numerator and denominator can scale arbitrarily while their ratio stays constant. Numerator and denominator orders `n` and `m` can in principle differ; for electrochemical systems `n = m` is the standard choice and the library default.

The model order must be chosen by the user, typically by sweeping over several values and comparing fit statistics (chi-square, AIC, BIC).

**Why LASSO regularization?** Without it, the numerator and denominator coefficients tend to compensate each other, growing to arbitrarily large values while the ratio remains finite. L1 regularization penalizes coefficient magnitudes and keeps the optimization well-conditioned. The regularization strength `reg_factor` must also be chosen by the user by comparing statistics. Setting `reg_factor=0` disables regularization and will typically cause the optimization to diverge.

> **Frequency convention** — all functions in this library expect **angular frequency ω (rad/s)**.
> Instruments typically report frequency in Hz; convert once before passing data to any library function:
> `omega = 2 * np.pi * freq_hz`.

---

## Features

- Rational polynomial models of any order, with independent numerator and denominator orders
- L1-regularized least-squares fitting via `lmfit`
- Four fitting strategies: single spectrum, multi-start, sequential, simultaneous
- Structured save/load for all fit types, with `metadata.json` for session identification
- Statistical tools for model selection on single and time-series datasets
- Visualization: Nyquist plots, residual plots, interactive spectrum slider, parameter evolution

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
from frf_lasso import fit_single, make_lmfit_model
from frf_lasso.io import save_single, load_single
from frf_lasso.visualization import nyquist_plot

# --- Load your data (format is up to you) ---
freq_hz   = np.load("frequencies.npy")            # shape: (N,), in Hz
omega     = 2 * np.pi * freq_hz                   # convert to rad/s — required by frf_lasso
impedance = np.load("impedance.npy")              # shape: (N,), complex

# --- Create model and initial parameters ---
model  = make_lmfit_model(num_order=6)
params = model.make_params()
for name in model.param_names:
    params[name].set(value=1.0, min=1e-9, max=1e6)

# --- Supply weighting factors ---
weights = 1 / np.abs(impedance) ** 0.5            # modulus weighting

# --- Fit ---
result, fit = fit_single(omega, impedance, model, params, weights, reg_factor=1e-8)
print(result.fit_report())

# --- Save ---
save_single("./results/my_fit", omega, impedance, model, result, fit,
            weights, reg_factor=1e-8)

# --- Visualize ---
nyquist_plot(impedance, fit=fit)
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

Neither `num_order` nor `reg_factor` has a universally correct value. The typical workflow is to sweep over both and compare the fit statistics:

```python
from frf_lasso import fit_single, make_lmfit_model
from frf_lasso.statistics import compare_fits

candidates = []
for order in [4, 5, 6, 7, 8]:
    for reg in [1e-10, 1e-8, 1e-6]:
        model  = make_lmfit_model(num_order=order)
        params = model.make_params()
        for name in model.param_names:
            params[name].set(value=1.0, min=1e-9, max=1e6)
        result, fit = fit_single(omega, impedance, model, params, weights, reg_factor=reg)
        candidates.append((order, reg, result))

compare_fits(candidates)   # prints a table sorted by AIC
```

For time-series datasets (sequential fit), use `compare_sequential_fits` instead, which aggregates per-spectrum statistics across all spectra:

```python
from frf_lasso.statistics import compare_sequential_fits

candidates = []
for order in [4, 5, 6, 7]:
    for reg in [1e-10, 1e-8, 1e-6]:
        model   = make_lmfit_model(num_order=order)
        params  = model.make_params()
        for name in model.param_names:
            params[name].set(value=1.0, min=1e-9, max=1e6)
        results, fits = fit_sequential(omega, impedance_set, model,
                                       params, weights, reg_factor=reg)
        candidates.append((order, reg, results))

compare_sequential_fits(candidates)   # prints cumulative, mean±std, median+IQR
```

A good fit has low chi-square and AIC without requiring an unnecessarily high polynomial order. If chi-square keeps improving as `reg_factor` approaches zero and the fit becomes visually poor, the regularization is doing its job.

---

## Fitting strategies

### 1. Single spectrum

Fits one impedance spectrum from a given starting point using `least_squares` optimization in log parameter space.

```python
model  = make_lmfit_model(num_order=6)
params = model.make_params()
for name in model.param_names:
    params[name].set(value=1.0, min=1e-9, max=1e6)

result, fit = fit_single(omega, impedance, model, params, weights, reg_factor=1e-8)
```

### 2. Multi-start

Repeats `fit_single` `n_starts` times with log-uniformly distributed random initial parameters. All results are returned so the caller can assess whether the optimization landscape has a well-defined global minimum.

```python
from frf_lasso import fit_multistart
from frf_lasso.statistics import multistart_statistics
from frf_lasso.visualization import multistart_plot, print_multistart_summary

results, fits = fit_multistart(
    omega, impedance, model, weights,
    n_starts=50, reg_factor=1e-8,
    param_min=1e-9, param_max=1e6,
    seed=42,
)

stats = multistart_statistics(results)
print_multistart_summary(stats)
multistart_plot(stats)
```

### 3. Sequential fit

Fits a time-ordered collection of spectra one by one, using the previous result as the warm start for the next. Suitable for datasets where impedance evolves slowly over time (e.g. battery cycling).

```python
from frf_lasso import fit_sequential

# impedance_set: shape (N_freq, T), complex — one spectrum per column
results, fits = fit_sequential(
    omega, impedance_set, model, params,
    weights,                        # (N,) or (N, T)
    reg_factor=1e-8,
)
```

### 4. Simultaneous fit

Jointly optimizes all spectra in a single minimization run with temporal smoothness penalties on parameter evolution. See [Notes on the simultaneous fit](#notes-on-the-simultaneous-fit) for the output structure and statistics caveat.

```python
from frf_lasso import fit_simultaneous

# seq_results: output of fit_sequential, used to initialize global parameters
global_result, results, fits = fit_simultaneous(
    omega, impedance_set, model, seq_results,
    weights, reg_factor=1e-8, smt_factor=1e-3,
)
```

---

## Saving and loading results

Each fit session is saved as a folder. The model is **not** serialized as a binary file — instead, `num_order` and `den_order` are stored in `metadata.json` and the model is recreated with `make_lmfit_model` on load. This keeps the saved data fully human-readable and avoids serialization fragility.

### Folder layouts

**Single:**
```
my_session/
  metadata.json      ← fit_type, num_order, den_order, reg_factor, chisqr, aic, bic
  omega.npy          ← angular frequencies (rad/s)
  impedance.npy      ← complex impedance
  weights.npy        ← weighting factors
  result.json        ← full lmfit result (params, statistics)
  fit.npy            ← fitted impedance
```

**Multi-start and sequential:**
```
my_session/
  metadata.json
  omega.npy
  impedance(_set).npy
  weights.npy
  results/
    result_000.json
    result_001.json  ...
  fits/
    fit_000.npy
    fit_001.npy      ...
```

**Simultaneous:**
```
my_session/
  metadata.json      ← includes stats_source: "global"
  omega.npy
  impedance_set.npy
  weights.npy
  global_result.json ← joint optimization result (authoritative statistics)
  results/           ← per-spectrum results (statistics inherited from global)
    result_000.json  ...
  fits/
    fit_000.npy      ...
```

### Save/load API

| Fit type | Save | Load returns |
|---|---|---|
| Single | `save_single(path, omega, impedance, model, result, fit, weights, reg_factor)` | `omega, impedance, model, result, fit, weights, metadata` |
| Multi-start | `save_multistart(path, omega, impedance, model, results, fits, weights, reg_factor)` | `omega, impedance, model, results, fits, weights, metadata` |
| Sequential | `save_sequential(path, omega, impedance_set, model, results, fits, weights, reg_factor)` | `omega, impedance_set, model, results, fits, weights, metadata` |
| Simultaneous | `save_simultaneous(path, omega, impedance_set, model, global_result, results, fits, weights, reg_factor, smt_factor)` | `omega, impedance_set, model, global_result, results, fits, weights, metadata` |

```python
from frf_lasso.io import save_sequential, load_sequential

save_sequential("./results/cycle1", omega, impedance_set, model,
                results, fits, weights, reg_factor=1e-8)

# In a later session:
omega, impedance_set, model, results, fits, weights, meta = load_sequential("./results/cycle1")
print(meta["reg_factor"], meta["num_order"])
```

---

## Statistics

```python
from frf_lasso.statistics import (
    compare_fits,             # single-spectrum model selection table
    compare_sequential_fits,  # time-series model selection table (3 aggregations)
    multistart_statistics,    # chi2 distribution across random starts
    param_evolution,          # extract {param: values_across_time} from results list
    smoothness_metrics,       # gradient and curvature of parameter evolution
)
```

`compare_sequential_fits` reports three complementary aggregations of chi2, AIC, and BIC across all spectra:

| Metric | What it shows |
|---|---|
| **Cumulative** | Sum across all spectra — equivalent ranking to one joint fit |
| **Mean ± std** | Average fit quality and spread; sensitive to outlier spectra |
| **Median + IQR** | Robust aggregation; not pulled by individual bad spectra |

---

## Visualization

```python
from frf_lasso.visualization import (
    nyquist_plot,            # single Nyquist plot, optional fit overlay
    residual_plot,           # weighted real/imag residuals vs frequency
    slider_plot,             # interactive Nyquist with spectrum slider
    multistart_plot,         # chi2 distribution panels (histogram, box, sequence)
    print_multistart_summary,# formatted console report of multistart statistics
    param_evolution_plot,    # parameter evolution, supports multiple series
)
```

All non-interactive functions accept an optional `ax` argument for embedding in larger figures and return `(fig, axes)`. The interactive `slider_plot` returns widget objects that must be kept alive in the caller's scope.

```python
from frf_lasso.visualization import param_evolution_plot

# Compare sequential and simultaneous parameter evolution on the same axes
param_evolution_plot({
    "sequential":    seq_results,
    "simultaneous":  sim_results,
})
```

---

## Package structure

```
frf_lasso/
  __init__.py           fit_single, fit_multistart, fit_sequential,
                        fit_simultaneous, adapt_params, make_lmfit_model
  models.py             rational_poly(), make_lmfit_model()
  objective.py          single_spectrum_residuals(), simultaneous_residuals()
  transformations.py    to_log(), to_linear()  [internal]
  fitting.py            fit_single(), fit_multistart(), fit_sequential()
  simultaneous.py       fit_simultaneous(), adapt_params()
  io.py                 save_*/load_* for all four fit types
  statistics.py         compare_fits(), compare_sequential_fits(),
                        multistart_statistics(), param_evolution(),
                        smoothness_metrics()
  visualization.py      nyquist_plot(), residual_plot(), slider_plot(),
                        multistart_plot(), print_multistart_summary(),
                        param_evolution_plot()

scripts/
  fit_single_spectrum.py     Example: single fit, sweep over order and reg_factor
  fit_multistart.py          Example: multi-start with consistency analysis
  fit_sequential.py          Example: sequential fit across a time-series
  fit_simultaneous.py        Example: simultaneous fit (experimental)
```

---

## Design notes

**Log-space optimization** — fitting is performed in log parameter space internally. This gives the optimizer multiplicative steps rather than additive ones, which is better suited to parameters that span many orders of magnitude. The L1 penalty is also applied in log space, which penalizes the order of magnitude of each coefficient rather than its raw magnitude, giving balanced regularization across all parameters. Parameters are transformed back to linear space before being returned.

**Positive parameter constraint** — the log-space transformation requires all parameter values to be strictly positive. Initial parameters must have positive values (set `min > 0`). This is not a limitation for typical EIS rational polynomial models.

**Model not serialized** — saved sessions store `num_order` and `den_order` in `metadata.json`. The model is recreated with `make_lmfit_model` on load, rather than serializing the model object. This avoids binary serialization fragility and keeps saved sessions fully human-readable.

**Asymmetric polynomial orders** — `make_lmfit_model(num_order, den_order)` supports different numerator and denominator orders. `den_order` defaults to `num_order` when not specified, which is the standard choice for electrochemical systems.

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

`fit_simultaneous()` jointly optimizes all spectra in a time-series in a single optimization run, adding temporal smoothness penalties (1st and 2nd derivative of parameter evolution) to the objective. In practice this approach is significantly slower and did not yield better results than the sequential fit on the test datasets. It is included for completeness and experimentation.

### Output structure

Unlike the other fitting functions which return `(results, fits)`, `fit_simultaneous` returns a 3-tuple:

```python
global_result, results, fits = fit_simultaneous(...)
```

| Return value | Type | Description |
|---|---|---|
| `global_result` | `lmfit.MinimizerResult` | The raw output of the single joint optimisation. Parameters are named `{param}_t{t}` (e.g. `a0_t0`, `a0_t1`, ...). This is the only result that carries statistically meaningful fit statistics. |
| `results` | `list` of `lmfit.MinimizerResult` | Per-spectrum results with local parameter names (e.g. `a0`, `a1`, ...), structurally identical to the output of `fit_sequential`. |
| `fits` | `list` of `ndarray` | Per-spectrum model predictions, same as all other fitting functions. |

### Statistics caveat

The `chisqr`, `aic`, `bic` and related attributes in each element of `results` are **inherited from `global_result`** — they describe the quality of the full joint optimisation, not of any individual spectrum. Do not compare these values directly against per-spectrum statistics from `fit_single` or `fit_sequential`. Use `global_result` for any statistical assessment of the simultaneous fit.

---

## License

MIT
