"""
Simultaneous global fitting over a time-series of impedance spectra.

Unlike the sequential fit (which optimises one spectrum at a time),
``fit_simultaneous`` optimises all spectra in a single minimisation run,
penalising rapid changes in the fitted parameters across time.  This
encourages smooth, physically plausible parameter trajectories.

Typical workflow
----------------
1. Run ``fit_sequential`` to get a good initialisation for all spectra.
2. Pass the sequential results to ``fit_simultaneous`` to refine them
   jointly with temporal smoothness constraints.

    seq_results, seq_fits = fit_sequential(omega, impedance_set, model,
                                           params, weights, reg_factor)
    result, fits = fit_simultaneous(omega, impedance_set, model,
                                    seq_results, weights,
                                    reg_factor, smt_factor)

Note
----
In practice this approach was found to be significantly slower than
sequential fitting without consistently producing better results.  It is
included for completeness and experimentation.  The output is a single
lmfit.MinimizerResult whose parameter set covers all time points, named
``{param}_t{t}`` (e.g. ``a0_t0``, ``a0_t1``, ...).
"""

import copy

import numpy as np
import lmfit

from .objective import simultaneous_residuals
from .transformations import to_log, to_linear


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _create_global_params(
    seq_results: list,
    model: lmfit.Model,
) -> lmfit.Parameters:
    """
    Build a global Parameters object from a list of sequential results.

    Each parameter from the model gets one entry per time point, named
    ``{param}_t{t}``.  Values are taken from the sequential fit results
    and are in linear space.

    Parameters
    ----------
    seq_results : list of lmfit.MinimizerResult
        Output of ``fit_sequential``.
    model : lmfit.Model
        The same model used for sequential fitting.

    Returns
    -------
    lmfit.Parameters
        Global parameter set in linear space, ready to be log-transformed
        before passing to the optimiser.
    """
    global_params = lmfit.Parameters()
    for t, result in enumerate(seq_results):
        for name in model.param_names:
            value = result.params[name].value
            min_  = result.params[name].min
            max_  = result.params[name].max
            global_params.add(f"{name}_t{t}", value=value, min=min_, max=max_)
    return global_params


def _extract_individual_results(
    global_result: lmfit.MinimizerResult,
    model: lmfit.Model,
    omega: np.ndarray,
    n_spectra: int,
) -> tuple:
    """
    Split a global MinimizerResult into per-spectrum results and fits.

    Each individual result is a deep copy of the global result with its
    ``params`` replaced by the local parameters for that time point.
    All other attributes — ``chisqr``, ``aic``, ``bic``, ``nfev``, etc. —
    are inherited from the global optimisation and remain unchanged.

    This means the statistics in each individual result describe the full
    simultaneous fit, not the fit quality of that spectrum alone.  This
    must be taken into account when saving and comparing results.
    """
    individual_results, fits = [], []
    for t in range(n_spectra):
        local_params = lmfit.Parameters()
        for name in model.param_names:
            src = global_result.params[f"{name}_t{t}"]
            local_params.add(name, value=src.value, min=src.min, max=src.max)

        local_result = copy.deepcopy(global_result)
        local_result.params = local_params

        individual_results.append(local_result)
        fits.append(model.eval(local_params, freq=omega))

    return individual_results, fits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_simultaneous(
    omega: np.ndarray,
    impedance_set: np.ndarray,
    model: lmfit.Model,
    seq_results: list,
    weights: np.ndarray,
    reg_factor: float = 1e-8,
    smt_factor: float = 1e-3,
) -> tuple:
    """
    Jointly optimise all spectra in a time-series with temporal smoothness.

    Uses the sequential fit results as initialisation, then refines all
    parameters together in a single minimisation with two additional penalty
    terms that discourage rapid changes in parameter values across time:

    - First-order finite difference penalty (encourages smooth evolution)
    - Second-order finite difference penalty (discourages sharp curvature)

    Both penalties are weighted by ``smt_factor``.

    Parameters
    ----------
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance_set : ndarray, shape (N, T), complex
        Impedance spectra at T successive time points.
    model : lmfit.Model
        Model returned by ``make_lmfit_model``.
    seq_results : list of lmfit.MinimizerResult, length T
        Output of ``fit_sequential``, used to initialise the global
        parameter set.
    weights : ndarray, shape (N,), real
        Per-frequency weighting factors applied equally to all spectra.
    reg_factor : float
        L1 regularization strength.
    smt_factor : float
        Temporal smoothness penalty strength.  Higher values enforce
        smoother parameter evolution at the cost of fit accuracy.

    Returns
    -------
    global_result : lmfit.MinimizerResult
        Single global optimisation result.  Parameters are named
        ``{param}_t{t}`` and are in linear space.  The statistics
        (``chisqr``, ``aic``, ``bic``, ...) describe the full joint fit.
    results : list of lmfit.MinimizerResult, length T
        Per-spectrum results with local parameters in linear space.
        Statistics are inherited from ``global_result`` — they reflect
        the global optimisation, not individual spectrum quality.
    fits : list of ndarray, each shape (N,), complex
        Per-spectrum model predictions.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != impedance_set.shape[0]:
        raise ValueError(
            f"weights must be 1D with length {impedance_set.shape[0]}, "
            f"got shape {weights.shape}"
        )

    n_spectra = impedance_set.shape[1]
    data_list = [impedance_set[:, t] for t in range(n_spectra)]

    global_params = _create_global_params(seq_results, model)
    log_global_params = to_log(global_params)

    global_result = lmfit.minimize(
        simultaneous_residuals,
        log_global_params,
        args=(omega, data_list, model, weights, reg_factor, smt_factor),
        method="least_squares",
        max_nfev=50000,
    )

    global_result.params = to_linear(global_result.params)
    results, fits = _extract_individual_results(global_result, model, omega, n_spectra)
    return global_result, results, fits


def adapt_params(
    seq_results: list,
    model: lmfit.Model,
    n_spectra_new: int,
) -> lmfit.Parameters:
    """
    Adapt a global parameter set when the number of spectra changes.

    Useful when running iterative multi-cycle experiments where the number
    of spectra in each cycle differs from the previous one.  The adapted
    parameters can be passed to ``fit_sequential`` as an initialisation.

    - If ``n_spectra_new`` > current count: new time points are initialised
      by copying the parameters of the last available spectrum.
    - If ``n_spectra_new`` < current count: trailing time points are dropped.
    - If equal: a copy is returned unchanged.

    Parameters
    ----------
    seq_results : list of lmfit.MinimizerResult
        Sequential results from the previous cycle.
    model : lmfit.Model
        The same model used for fitting.
    n_spectra_new : int
        Number of spectra in the new cycle.

    Returns
    -------
    lmfit.Parameters
        Adapted parameter set in linear space, ready for use as ``params``
        in ``fit_sequential``.
    """
    n_spectra_old = len(seq_results)
    param_names = model.param_names

    adapted = lmfit.Parameters()
    for t in range(n_spectra_new):
        # Clamp to last available result if new cycle is longer
        source_t = min(t, n_spectra_old - 1)
        for name in param_names:
            p = seq_results[source_t].params[name]
            adapted.add(f"{name}_t{t}", value=p.value, min=p.min, max=p.max)

    return adapted
