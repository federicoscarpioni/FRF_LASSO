"""
Fitting routines for rational polynomial impedance models.

Three fitting strategies are provided, each building on the previous:

  fit_single      — fit one impedance spectrum from a given starting point
  fit_multistart  — repeat fit_single from N random starting points and
                    return all results for consistency analysis
  fit_sequential  — fit a time-ordered collection of spectra, warm-starting
                    each fit from the previous result

All three functions:
  - accept omega in rad/s (convert from Hz with ``omega = 2 * pi * freq_hz``)
  - accept impedance as complex-valued NumPy arrays
  - return standard lmfit.MinimizerResult objects (see lmfit documentation
    for the full interface: params, chisqr, aic, bic, fit_report(), ...)
  - return the fitted impedance as a complex array alongside the result

Weighting
---------
All functions require the caller to supply a weights array explicitly.
There are many valid weighting strategies (modulus, proportional, unit,
frequency-dependent, etc.) and the right choice is problem-dependent.
Weights must be a real-valued NumPy array:

  - 1D, shape (N,)    — same weights for every spectrum
  - 2D, shape (N, T)  — one weight vector per spectrum (fit_sequential only)


Parameter sign convention
-------------------------
The log-space optimisation used internally requires all parameter values to
be strictly positive.  Coefficients are therefore constrained to (param_min,
param_max) during fitting.  For typical EIS rational polynomial models this
is not a limitation in practice, but it means the model cannot represent
impedance functions that require negative coefficients.
"""

import numpy as np
import lmfit

from .models import make_lmfit_model
from .objective import single_spectrum_residuals
from .transformations import to_log, to_linear


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_random_params(
    model: lmfit.Model,
    param_min: float,
    param_max: float,
    rng: np.random.Generator,
) -> lmfit.Parameters:
    """Return a Parameters object with log-uniformly distributed values."""
    params = lmfit.Parameters()
    log_min, log_max = np.log(param_min), np.log(param_max)
    for name in model.param_names:
        value = np.exp(rng.uniform(log_min, log_max))
        params.add(name, value=value, min=param_min, max=param_max)
    return params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_single(
    omega: np.ndarray,
    impedance: np.ndarray,
    model: lmfit.Model,
    params: lmfit.Parameters,
    weights: np.ndarray,
    reg_factor: float = 1e-8,
) -> tuple:
    """
    Fit a single impedance spectrum to a rational polynomial model.

    Optimisation is performed in log parameter space (see transformations.py)
    using lmfit's least_squares method.  The result is converted back to
    linear space before being returned.

    Parameters
    ----------
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance : ndarray, shape (N,), complex
        Measured impedance values.
    model : lmfit.Model
        Model returned by ``make_lmfit_model``.
    params : lmfit.Parameters
        Initial parameter values in linear space.  All values must be
        strictly positive (see module docstring).
    weights : ndarray, shape (N,), real
        Per-frequency weighting factors.  See module docstring for
        weighting strategy guidance.
    reg_factor : float
        L1 regularization strength.  Controls the penalty on coefficient
        magnitudes in log space.  Set to 0 to disable (not recommended —
        the optimisation typically diverges without regularization).

    Returns
    -------
    result : lmfit.MinimizerResult
        Optimisation result with parameters in linear space.
    fit : ndarray, shape (N,), complex
        Model-predicted impedance at the fitted parameters.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != impedance.shape[:1]:
        raise ValueError(
            f"weights shape {weights.shape} does not match "
            f"number of frequencies {impedance.shape[0]}"
        )

    log_params = to_log(params)

    result = lmfit.minimize(
        single_spectrum_residuals,
        log_params,
        args=(omega, impedance, model, weights, reg_factor),
        method="least_squares",
        max_nfev=50000,
    )

    result.params = to_linear(result.params)
    fit = model.eval(result.params, freq=omega)
    return result, fit


def fit_multistart(
    omega: np.ndarray,
    impedance: np.ndarray,
    model: lmfit.Model,
    weights: np.ndarray,
    n_starts: int = 50,
    reg_factor: float = 1e-8,
    param_min: float = 1e-9,
    param_max: float = 1e6,
    seed: int = 42,
) -> tuple:
    """
    Fit a single impedance spectrum from multiple random starting points.

    Repeats ``fit_single`` ``n_starts`` times, each time drawing initial
    parameter values log-uniformly from (param_min, param_max).  All results
    are returned so that the caller can assess solution uniqueness and select
    the best fit (e.g. with ``statistics.consistency_metrics``).

    Parameters
    ----------
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance : ndarray, shape (N,), complex
        Measured impedance values.
    model : lmfit.Model
        Model returned by ``make_lmfit_model``.
    weights : ndarray, shape (N,), real
        Per-frequency weighting factors.  See module docstring for
        weighting strategy guidance.
    n_starts : int
        Number of random starting points.
    reg_factor : float
        L1 regularization strength.
    param_min : float
        Lower bound for random parameter initialisation (linear space).
    param_max : float
        Upper bound for random parameter initialisation (linear space).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    results : list of lmfit.MinimizerResult
        One result per starting point, in the order they were run.
    fits : list of ndarray, each shape (N,), complex
        Corresponding model predictions.
    """
    rng = np.random.default_rng(seed)
    results, fits = [], []

    for _ in range(n_starts):
        params = _make_random_params(model, param_min, param_max, rng)
        result, fit = fit_single(omega, impedance, model, params, weights, reg_factor)
        results.append(result)
        fits.append(fit)

    return results, fits


def fit_sequential(
    omega: np.ndarray,
    impedance_set: np.ndarray,
    model: lmfit.Model,
    params: lmfit.Parameters,
    weights: np.ndarray,
    reg_factor: float = 1e-8,
) -> tuple:
    """
    Fit a time-ordered sequence of impedance spectra sequentially.

    The first spectrum is fitted from the user-supplied ``params``.  Each
    subsequent spectrum is fitted using the previous result as the starting
    point, so that slowly-evolving parameter changes are tracked smoothly.

    Parameters
    ----------
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance_set : ndarray, shape (N, T), complex
        Impedance spectra at T successive time points.  Each column is one
        spectrum.
    model : lmfit.Model
        Model returned by ``make_lmfit_model``.
    params : lmfit.Parameters
        Initial parameter values in linear space for the first spectrum.
    weights : ndarray, shape (N,) or (N, T), real
        Per-frequency weighting factors.  Two forms are accepted:

        - 1D array, shape (N,) — the same weights are applied to every
          spectrum.
        - 2D array, shape (N, T) — column t is used for spectrum t.
          Required when using data-dependent strategies such as modulus
          weighting (``1 / |Z_t|^0.5``), since those vary per spectrum.

        See module docstring for weighting strategy guidance.
    reg_factor : float
        L1 regularization strength applied to every spectrum.

    Returns
    -------
    results : list of lmfit.MinimizerResult, length T
        One result per spectrum, in time order.
    fits : list of ndarray, each shape (N,), complex
        Corresponding model predictions.
    """
    n_freq, n_spectra = impedance_set.shape
    weights = np.asarray(weights, dtype=float)

    if weights.ndim == 1:
        if weights.shape[0] != n_freq:
            raise ValueError(
                f"1D weights length {weights.shape[0]} does not match "
                f"number of frequencies {n_freq}"
            )
    elif weights.ndim == 2:
        if weights.shape != (n_freq, n_spectra):
            raise ValueError(
                f"2D weights shape {weights.shape} does not match "
                f"impedance_set shape {impedance_set.shape}"
            )
    else:
        raise ValueError("weights must be a 1D or 2D array")

    results, fits = [], []
    current_params = params.copy()

    for t in range(n_spectra):
        impedance = impedance_set[:, t]
        w = weights if weights.ndim == 1 else weights[:, t]
        result, fit = fit_single(omega, impedance, model, current_params, w, reg_factor)
        results.append(result)
        fits.append(fit)
        current_params = result.params.copy()

    return results, fits
