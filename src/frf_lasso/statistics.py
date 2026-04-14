"""
Statistical analysis of fitting results.

Functions are grouped by use case:

  Model selection (single spectrum)
    compare_fits               — tabulate chi2/AIC/BIC across orders and reg_factors

  Model selection (sequential / simultaneous)
    compare_sequential_fits    — aggregate chi2/AIC/BIC across spectra and compare
                                 candidates using three complementary metrics:
                                 cumulative sum, mean ± std, and median + IQR

  Multi-start consistency
    multistart_statistics      — distribution of chi2 across random starts

  Time-series analysis (sequential / simultaneous)
    param_evolution            — extract parameter values across time
    smoothness_metrics         — quantify how smoothly parameters evolve
"""

import numpy as np
import lmfit


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def compare_fits(candidates: list) -> list:
    """
    Compare fitting results across different model orders and regularization
    factors for a single spectrum.

    Parameters
    ----------
    candidates : list of (num_order, reg_factor, result) tuples
        Each entry corresponds to one fit attempt.  ``result`` is a
        ``lmfit.MinimizerResult``.  Fits are sorted by AIC (ascending).

    Returns
    -------
    rows : list of dict
        One dict per candidate with keys: ``num_order``, ``reg_factor``,
        ``n_params``, ``chisqr``, ``redchi``, ``aic``, ``bic``.
        Sorted by AIC ascending.

    Examples
    --------
    >>> candidates = []
    >>> for order in [4, 5, 6, 7]:
    ...     for reg in [1e-10, 1e-8, 1e-6]:
    ...         result, fit = fit_single(omega, impedance, model, params,
    ...                                  weights, reg_factor=reg)
    ...         candidates.append((order, reg, result))
    >>> rows = compare_fits(candidates)
    """
    rows = []
    for num_order, reg_factor, result in candidates:
        rows.append({
            "num_order":  num_order,
            "reg_factor": reg_factor,
            "n_params":   result.nvarys,
            "chisqr":     result.chisqr,
            "redchi":     result.redchi,
            "aic":        result.aic,
            "bic":        result.bic,
        })

    rows.sort(key=lambda r: r["aic"])

    # Print formatted table
    header = (f"{'order':>6}  {'reg_factor':>12}  {'n_params':>8}  "
              f"{'chisqr':>12}  {'redchi':>10}  {'AIC':>12}  {'BIC':>12}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['num_order']:>6}  {r['reg_factor']:>12.2e}  {r['n_params']:>8}  "
              f"{r['chisqr']:>12.4e}  {r['redchi']:>10.4e}  "
              f"{r['aic']:>12.4e}  {r['bic']:>12.4e}")

    return rows


# ---------------------------------------------------------------------------
# Model selection — sequential / simultaneous
# ---------------------------------------------------------------------------

def compare_sequential_fits(candidates: list) -> list:
    """
    Compare sequential (or simultaneous) fit candidates across multiple spectra.

    When fitting a time-series, each candidate configuration (polynomial order,
    regularization factor) produces T results — one per spectrum — each with its
    own chi2, AIC, and BIC.  This function aggregates those per-spectrum values
    into three complementary summary metrics for each candidate:

    - **Cumulative** — sum across all spectra.  Equivalent to treating the
      full time-series as one joint fit; gives a single clean ranking number.
    - **Mean ± std** — standard average and spread.  Intuitive but sensitive
      to outlier spectra that happen to fit poorly.
    - **Median + IQR** — robust aggregation.  The median reflects typical
      fit quality; the IQR quantifies spread without being pulled by outliers.

    All three metrics are reported for chi2, AIC, and BIC, printed as three
    consecutive tables sorted by cumulative chi2 (ascending).

    Parameters
    ----------
    candidates : list of (num_order, reg_factor, results) tuples
        ``results`` is the list of ``lmfit.MinimizerResult`` returned by
        ``fit_sequential`` or the per-spectrum results from ``fit_simultaneous``.

    Returns
    -------
    rows : list of dict
        One dict per candidate, sorted by cumulative chi2 ascending.  Each
        dict contains all computed aggregations for chi2, AIC, and BIC.

    Examples
    --------
    >>> candidates = []
    >>> for order in [4, 5, 6, 7]:
    ...     for reg in [1e-10, 1e-8, 1e-6]:
    ...         results, fits = fit_sequential(omega, impedance_set, model,
    ...                                        params, weights, reg_factor=reg)
    ...         candidates.append((order, reg, results))
    >>> rows = compare_sequential_fits(candidates)
    """
    def _agg(values: np.ndarray) -> dict:
        return {
            "cumulative": float(values.sum()),
            "mean":       float(values.mean()),
            "std":        float(values.std()),
            "median":     float(np.median(values)),
            "iqr":        float(np.percentile(values, 75) - np.percentile(values, 25)),
        }

    rows = []
    for num_order, reg_factor, results in candidates:
        chi2 = np.array([r.chisqr for r in results])
        aic  = np.array([r.aic   for r in results])
        bic  = np.array([r.bic   for r in results])
        rows.append({
            "num_order":  num_order,
            "reg_factor": reg_factor,
            "n_spectra":  len(results),
            "chi2": _agg(chi2),
            "aic":  _agg(aic),
            "bic":  _agg(bic),
        })

    rows.sort(key=lambda r: r["chi2"]["cumulative"])

    # --- printing ---
    def _header(metric: str) -> str:
        return (f"\n{'order':>6}  {'reg_factor':>12}  "
                f"{metric + '_cumul':>14}  "
                f"{metric + '_mean':>13}  {'±std':>11}  "
                f"{metric + '_median':>14}  {'IQR':>11}")

    def _row(r: dict, metric: str) -> str:
        m = r[metric]
        return (f"{r['num_order']:>6}  {r['reg_factor']:>12.2e}  "
                f"{m['cumulative']:>14.4e}  "
                f"{m['mean']:>13.4e}  {m['std']:>11.4e}  "
                f"{m['median']:>14.4e}  {m['iqr']:>11.4e}")

    for metric in ("chi2", "aic", "bic"):
        header = _header(metric.upper())
        print(header)
        print("-" * len(header))
        for r in rows:
            print(_row(r, metric))

    return rows


# ---------------------------------------------------------------------------
# Multi-start consistency
# ---------------------------------------------------------------------------

def multistart_statistics(results: list) -> dict:
    """
    Compute chi-square statistics across a set of multi-start fit results.

    Useful for assessing whether the optimisation landscape has a well-defined
    global minimum.  A high coefficient of variation or low consistency ratio
    indicates multiple local minima or a poorly conditioned objective.

    Parameters
    ----------
    results : list of lmfit.MinimizerResult
        Output of ``fit_multistart``.

    Returns
    -------
    stats : dict
        Keys:
        ``n_starts``          — number of trials
        ``chi2_values``       — array of chi-square values, one per trial
        ``mean``, ``std``     — mean and standard deviation of chi2
        ``median``            — median chi2
        ``min``, ``max``      — best and worst chi2
        ``iqr``               — interquartile range of chi2
        ``cv``                — coefficient of variation (std / mean)
        ``consistency_ratio`` — fraction of trials within 2× the best chi2
        ``best_index``        — index of the trial with the lowest chi2
    """
    chi2 = np.array([r.chisqr for r in results])
    min_chi2 = chi2.min()

    return {
        "n_starts":          len(results),
        "chi2_values":       chi2,
        "mean":              chi2.mean(),
        "std":               chi2.std(),
        "median":            np.median(chi2),
        "min":               min_chi2,
        "max":               chi2.max(),
        "iqr":               np.percentile(chi2, 75) - np.percentile(chi2, 25),
        "cv":                chi2.std() / chi2.mean() if chi2.mean() != 0 else np.inf,
        "consistency_ratio": np.mean(chi2 <= 2 * min_chi2),
        "best_index":        int(np.argmin(chi2)),
    }


# ---------------------------------------------------------------------------
# Time-series analysis
# ---------------------------------------------------------------------------

def param_evolution(results: list) -> dict:
    """
    Extract the evolution of each parameter across a sequence of spectra.

    Works with the output of both ``fit_sequential`` and the per-spectrum
    ``results`` list returned by ``fit_simultaneous``.

    Parameters
    ----------
    results : list of lmfit.MinimizerResult, length T
        Per-spectrum fitting results in time order.

    Returns
    -------
    evolution : dict
        Keys are parameter names; values are 1D arrays of length T
        containing the fitted parameter value at each time point.

    Examples
    --------
    >>> evo = param_evolution(seq_results)
    >>> plt.plot(evo['a0'])
    """
    param_names = list(results[0].params.keys())
    return {
        name: np.array([r.params[name].value for r in results])
        for name in param_names
    }


def smoothness_metrics(results: list) -> dict:
    """
    Quantify how smoothly each parameter evolves across a sequence of spectra.

    Two metrics are computed per parameter:

    - ``gradient``  — sum of squared first differences (penalises rapid change)
    - ``curvature`` — sum of squared second differences (penalises sharp turns)

    Lower values indicate smoother evolution.  These metrics can be used to
    compare sequential and simultaneous fits: a simultaneous fit with a
    non-zero ``smt_factor`` should produce lower curvature than the
    sequential fit.

    Parameters
    ----------
    results : list of lmfit.MinimizerResult, length T
        Per-spectrum fitting results in time order.

    Returns
    -------
    metrics : dict
        Keys are parameter names; values are dicts with keys
        ``gradient`` (float) and ``curvature`` (float).
    """
    evo = param_evolution(results)
    metrics = {}

    for name, values in evo.items():
        first_diff  = np.diff(values)
        second_diff = values[2:] - 2 * values[1:-1] + values[:-2]
        metrics[name] = {
            "gradient":  float(np.sum(first_diff  ** 2)) if len(values) > 1 else 0.0,
            "curvature": float(np.sum(second_diff ** 2)) if len(values) > 2 else 0.0,
        }

    return metrics
