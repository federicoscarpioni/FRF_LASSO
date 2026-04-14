"""
Save and load fitting sessions to/from disk.

Each session is stored as a folder containing:

  metadata.json      — fit type, model orders, regularization parameters,
                       and summary statistics.  Readable without loading
                       any arrays; useful for scanning a directory of saved
                       sessions.
  omega.npy          — angular frequencies (rad/s)
  impedance(_set).npy — original complex impedance data
  weights.npy        — weighting factors used during fitting
  result(s)          — lmfit result(s) serialised as JSON

The model is NOT saved as a binary file.  Instead, ``num_order`` and
``den_order`` are stored in metadata and the model is recreated on load
with ``make_lmfit_model``.  This avoids dill-serialisation fragility for
the closure-based model and keeps the saved data fully human-readable.

Folder layouts
--------------
Single:
    metadata.json  omega.npy  impedance.npy  weights.npy
    result.json  fit.npy

Multistart:
    metadata.json  omega.npy  impedance.npy  weights.npy
    results/result_000.json … result_N.json
    fits/fit_000.npy … fit_N.npy

Sequential:
    metadata.json  omega.npy  impedance_set.npy  weights.npy
    results/result_000.json … result_T.json
    fits/fit_000.npy … fit_T.npy

Simultaneous:
    metadata.json  omega.npy  impedance_set.npy  weights.npy
    global_result.json
    results/result_000.json … result_T.json
    fits/fit_000.npy … fit_T.npy
"""

import json
import os

import numpy as np
import lmfit

from .models import make_lmfit_model


# ---------------------------------------------------------------------------
# Internal helpers — result serialisation
# ---------------------------------------------------------------------------

def _result_to_dict(result: lmfit.MinimizerResult) -> dict:
    """Serialise a MinimizerResult to a JSON-compatible dictionary."""
    return {
        "method":    result.method,
        "success":   result.success,
        "message":   result.message,
        "nfev":      result.nfev,
        "ndata":     result.ndata,
        "nvarys":    result.nvarys,
        "nfree":     result.nfree,
        "chisqr":    result.chisqr,
        "redchi":    result.redchi,
        "aic":       result.aic,
        "bic":       result.bic,
        "errorbars": result.errorbars,
        "params": {
            name: {
                "value":  p.value,
                "stderr": p.stderr,
                "vary":   p.vary,
                "min":    p.min,
                "max":    p.max,
                "expr":   p.expr,
            }
            for name, p in result.params.items()
        },
    }


def _dict_to_result(d: dict) -> lmfit.MinimizerResult:
    """Reconstruct a MinimizerResult from a serialised dictionary."""
    params = lmfit.Parameters()
    for name, pd in d["params"].items():
        params.add(name, value=pd["value"], vary=pd["vary"],
                   min=pd["min"], max=pd["max"], expr=pd["expr"])
        if pd["stderr"] is not None:
            params[name].stderr = pd["stderr"]

    result = lmfit.MinimizerResult()
    result.params    = params
    result.method    = d["method"]
    result.success   = d["success"]
    result.message   = d["message"]
    result.nfev      = d["nfev"]
    result.ndata     = d["ndata"]
    result.nvarys    = d["nvarys"]
    result.nfree     = d["nfree"]
    result.chisqr    = d["chisqr"]
    result.redchi    = d["redchi"]
    result.aic       = d["aic"]
    result.bic       = d["bic"]
    result.errorbars = d["errorbars"]
    return result


# ---------------------------------------------------------------------------
# Internal helpers — folder I/O
# ---------------------------------------------------------------------------

def _extract_orders(model: lmfit.Model) -> tuple:
    """Return (num_order, den_order) inferred from the model's parameter names."""
    a_params = [n for n in model.param_names if n.startswith("a")]
    b_params = [n for n in model.param_names if n.startswith("b")]
    return len(a_params) - 1, len(b_params)


def _save_result(path: str, result: lmfit.MinimizerResult, filename: str = "result.json"):
    with open(os.path.join(path, filename), "w") as f:
        json.dump(_result_to_dict(result), f, indent=2)


def _load_result(path: str, filename: str = "result.json") -> lmfit.MinimizerResult:
    with open(os.path.join(path, filename)) as f:
        return _dict_to_result(json.load(f))


def _save_results_list(path: str, results: list, fits: list):
    """Save a list of results and fits into results/ and fits/ subfolders."""
    res_dir = os.path.join(path, "results")
    fit_dir = os.path.join(path, "fits")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(fit_dir, exist_ok=True)
    for i, (result, fit) in enumerate(zip(results, fits)):
        _save_result(res_dir, result, f"result_{i:03d}.json")
        np.save(os.path.join(fit_dir, f"fit_{i:03d}.npy"), fit)


def _load_results_list(path: str) -> tuple:
    """Load results/ and fits/ subfolders into lists."""
    res_dir = os.path.join(path, "results")
    fit_dir = os.path.join(path, "fits")
    files = sorted(f for f in os.listdir(res_dir) if f.endswith(".json"))
    results, fits = [], []
    for fname in files:
        results.append(_load_result(res_dir, fname))
        fit_fname = fname.replace("result_", "fit_").replace(".json", ".npy")
        fits.append(np.load(os.path.join(fit_dir, fit_fname)))
    return results, fits


def _save_metadata(path: str, meta: dict):
    with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_metadata(path: str) -> dict:
    with open(os.path.join(path, "metadata.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public API — save
# ---------------------------------------------------------------------------

def save_single(
    path: str,
    omega: np.ndarray,
    impedance: np.ndarray,
    model: lmfit.Model,
    result: lmfit.MinimizerResult,
    fit: np.ndarray,
    weights: np.ndarray,
    reg_factor: float,
):
    """
    Save a single-spectrum fit session.

    Parameters
    ----------
    path : str
        Destination folder (created if it does not exist).
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance : ndarray, shape (N,), complex
        Original measured impedance.
    model : lmfit.Model
        Model used for fitting (orders are extracted and stored in metadata).
    result : lmfit.MinimizerResult
        Output of ``fit_single``.
    fit : ndarray, shape (N,), complex
        Fitted impedance values.
    weights : ndarray, shape (N,), real
        Weighting factors used during fitting.
    reg_factor : float
        L1 regularization strength used during fitting.
    """
    os.makedirs(path, exist_ok=True)
    num_order, den_order = _extract_orders(model)

    _save_metadata(path, {
        "fit_type":   "single",
        "num_order":  num_order,
        "den_order":  den_order,
        "reg_factor": reg_factor,
        "n_freq":     len(omega),
        "chisqr":     result.chisqr,
        "aic":        result.aic,
        "bic":        result.bic,
    })
    np.save(os.path.join(path, "omega.npy"),     omega)
    np.save(os.path.join(path, "impedance.npy"), impedance)
    np.save(os.path.join(path, "weights.npy"),   weights)
    np.save(os.path.join(path, "fit.npy"),       fit)
    _save_result(path, result)


def save_multistart(
    path: str,
    omega: np.ndarray,
    impedance: np.ndarray,
    model: lmfit.Model,
    results: list,
    fits: list,
    weights: np.ndarray,
    reg_factor: float,
):
    """
    Save a multi-start fit session.

    Parameters
    ----------
    path : str
        Destination folder (created if it does not exist).
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance : ndarray, shape (N,), complex
        Original measured impedance.
    model : lmfit.Model
        Model used for fitting.
    results : list of lmfit.MinimizerResult
        Output of ``fit_multistart``.
    fits : list of ndarray, each shape (N,), complex
        Fitted impedance values for each start.
    weights : ndarray, shape (N,), real
        Weighting factors used during fitting.
    reg_factor : float
        L1 regularization strength used during fitting.
    """
    os.makedirs(path, exist_ok=True)
    num_order, den_order = _extract_orders(model)
    best_chisqr = min(r.chisqr for r in results)

    _save_metadata(path, {
        "fit_type":    "multistart",
        "num_order":   num_order,
        "den_order":   den_order,
        "reg_factor":  reg_factor,
        "n_freq":      len(omega),
        "n_starts":    len(results),
        "best_chisqr": best_chisqr,
    })
    np.save(os.path.join(path, "omega.npy"),     omega)
    np.save(os.path.join(path, "impedance.npy"), impedance)
    np.save(os.path.join(path, "weights.npy"),   weights)
    _save_results_list(path, results, fits)


def save_sequential(
    path: str,
    omega: np.ndarray,
    impedance_set: np.ndarray,
    model: lmfit.Model,
    results: list,
    fits: list,
    weights: np.ndarray,
    reg_factor: float,
):
    """
    Save a sequential fit session.

    Parameters
    ----------
    path : str
        Destination folder (created if it does not exist).
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance_set : ndarray, shape (N, T), complex
        Original measured impedance at each time point.
    model : lmfit.Model
        Model used for fitting.
    results : list of lmfit.MinimizerResult, length T
        Output of ``fit_sequential``.
    fits : list of ndarray, each shape (N,), complex
        Fitted impedance values for each spectrum.
    weights : ndarray, shape (N,) or (N, T), real
        Weighting factors used during fitting.
    reg_factor : float
        L1 regularization strength used during fitting.
    """
    os.makedirs(path, exist_ok=True)
    num_order, den_order = _extract_orders(model)
    n_freq, n_spectra = impedance_set.shape

    _save_metadata(path, {
        "fit_type":  "sequential",
        "num_order": num_order,
        "den_order": den_order,
        "reg_factor": reg_factor,
        "n_freq":    n_freq,
        "n_spectra": n_spectra,
    })
    np.save(os.path.join(path, "omega.npy"),         omega)
    np.save(os.path.join(path, "impedance_set.npy"), impedance_set)
    np.save(os.path.join(path, "weights.npy"),       weights)
    _save_results_list(path, results, fits)


def save_simultaneous(
    path: str,
    omega: np.ndarray,
    impedance_set: np.ndarray,
    model: lmfit.Model,
    global_result: lmfit.MinimizerResult,
    results: list,
    fits: list,
    weights: np.ndarray,
    reg_factor: float,
    smt_factor: float,
):
    """
    Save a simultaneous fit session.

    Both the global result (single joint optimisation) and the per-spectrum
    results are saved.  ``metadata.json`` carries ``stats_source: "global"``
    to flag that per-spectrum statistics are inherited from the global fit.

    Parameters
    ----------
    path : str
        Destination folder (created if it does not exist).
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    impedance_set : ndarray, shape (N, T), complex
        Original measured impedance at each time point.
    model : lmfit.Model
        Model used for fitting.
    global_result : lmfit.MinimizerResult
        First element of the tuple returned by ``fit_simultaneous``.
        Contains the joint optimisation statistics and all ``{param}_t{t}``
        parameters.
    results : list of lmfit.MinimizerResult, length T
        Second element of the tuple returned by ``fit_simultaneous``.
        Per-spectrum results with local parameter names; statistics
        inherited from ``global_result``.
    fits : list of ndarray, each shape (N,), complex
        Per-spectrum model predictions.
    weights : ndarray, shape (N,), real
        Weighting factors used during fitting.
    reg_factor : float
        L1 regularization strength used during fitting.
    smt_factor : float
        Temporal smoothness penalty strength used during fitting.
    """
    os.makedirs(path, exist_ok=True)
    num_order, den_order = _extract_orders(model)
    n_freq, n_spectra = impedance_set.shape

    _save_metadata(path, {
        "fit_type":    "simultaneous",
        "num_order":   num_order,
        "den_order":   den_order,
        "reg_factor":  reg_factor,
        "smt_factor":  smt_factor,
        "n_freq":      n_freq,
        "n_spectra":   n_spectra,
        "stats_source": "global",
        "chisqr":      global_result.chisqr,
        "aic":         global_result.aic,
        "bic":         global_result.bic,
    })
    np.save(os.path.join(path, "omega.npy"),         omega)
    np.save(os.path.join(path, "impedance_set.npy"), impedance_set)
    np.save(os.path.join(path, "weights.npy"),       weights)
    _save_result(path, global_result, "global_result.json")
    _save_results_list(path, results, fits)


# ---------------------------------------------------------------------------
# Public API — load
# ---------------------------------------------------------------------------

def load_single(path: str) -> tuple:
    """
    Load a single-spectrum fit session.

    Returns
    -------
    omega : ndarray, shape (N,)
    impedance : ndarray, shape (N,), complex
    model : lmfit.Model
    result : lmfit.MinimizerResult
    fit : ndarray, shape (N,), complex
    weights : ndarray, shape (N,)
    metadata : dict
    """
    meta      = _load_metadata(path)
    model     = make_lmfit_model(meta["num_order"], meta["den_order"])
    omega     = np.load(os.path.join(path, "omega.npy"))
    impedance = np.load(os.path.join(path, "impedance.npy"))
    weights   = np.load(os.path.join(path, "weights.npy"))
    fit       = np.load(os.path.join(path, "fit.npy"))
    result    = _load_result(path)
    return omega, impedance, model, result, fit, weights, meta


def load_multistart(path: str) -> tuple:
    """
    Load a multi-start fit session.

    Returns
    -------
    omega : ndarray, shape (N,)
    impedance : ndarray, shape (N,), complex
    model : lmfit.Model
    results : list of lmfit.MinimizerResult
    fits : list of ndarray, each shape (N,), complex
    weights : ndarray, shape (N,)
    metadata : dict
    """
    meta      = _load_metadata(path)
    model     = make_lmfit_model(meta["num_order"], meta["den_order"])
    omega     = np.load(os.path.join(path, "omega.npy"))
    impedance = np.load(os.path.join(path, "impedance.npy"))
    weights   = np.load(os.path.join(path, "weights.npy"))
    results, fits = _load_results_list(path)
    return omega, impedance, model, results, fits, weights, meta


def load_sequential(path: str) -> tuple:
    """
    Load a sequential fit session.

    Returns
    -------
    omega : ndarray, shape (N,)
    impedance_set : ndarray, shape (N, T), complex
    model : lmfit.Model
    results : list of lmfit.MinimizerResult, length T
    fits : list of ndarray, each shape (N,), complex
    weights : ndarray, shape (N,) or (N, T)
    metadata : dict
    """
    meta          = _load_metadata(path)
    model         = make_lmfit_model(meta["num_order"], meta["den_order"])
    omega         = np.load(os.path.join(path, "omega.npy"))
    impedance_set = np.load(os.path.join(path, "impedance_set.npy"))
    weights       = np.load(os.path.join(path, "weights.npy"))
    results, fits = _load_results_list(path)
    return omega, impedance_set, model, results, fits, weights, meta


def load_simultaneous(path: str) -> tuple:
    """
    Load a simultaneous fit session.

    The global result contains the joint optimisation statistics and all
    ``{param}_t{t}`` parameters.  The per-spectrum results carry local
    parameter names but their statistics are inherited from the global fit
    (see ``metadata["stats_source"]``).

    Returns
    -------
    omega : ndarray, shape (N,)
    impedance_set : ndarray, shape (N, T), complex
    model : lmfit.Model
    global_result : lmfit.MinimizerResult
    results : list of lmfit.MinimizerResult, length T
    fits : list of ndarray, each shape (N,), complex
    weights : ndarray, shape (N,)
    metadata : dict
    """
    meta          = _load_metadata(path)
    model         = make_lmfit_model(meta["num_order"], meta["den_order"])
    omega         = np.load(os.path.join(path, "omega.npy"))
    impedance_set = np.load(os.path.join(path, "impedance_set.npy"))
    weights       = np.load(os.path.join(path, "weights.npy"))
    global_result = _load_result(path, "global_result.json")
    results, fits = _load_results_list(path)
    return omega, impedance_set, model, global_result, results, fits, weights, meta
