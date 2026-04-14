"""
Objective (residual) functions for rational polynomial fitting.

Both functions return a flat real-valued residual vector suitable for
lmfit.minimize (which uses least-squares internally and expects real
residuals).  The vector is assembled from three concatenated blocks:

    [ fitting residuals | L1 penalty | temporal penalties (optional) ]

**Fitting residuals**
    Real and imaginary parts of the weighted model error are stacked:
    [w*(Z_model.real - Z_data.real), w*(Z_model.imag - Z_data.imag)].
    Splitting complex residuals into two real parts is required because
    least-squares optimisers work in real space.

**L1 regularization**
    The penalty block is reg_factor * log_params_values, where log_params
    are the parameter values *in log space* (as passed by fitting.py).
    Penalising the log values shrinks the order of magnitude of each
    coefficient rather than its raw magnitude, giving balanced
    regularisation across parameters that differ by many decades.
    Setting reg_factor=0 disables regularisation, which will typically
    cause the numerator and denominator coefficients to grow unboundedly.

**Temporal penalties (simultaneous fit only)**
    First- and second-order finite differences of each parameter across
    time are appended to penalise rapid changes in parameter evolution.
    Controlled by smt_factor.
"""

import numpy as np
import lmfit

from .transformations import to_linear


def single_spectrum_residuals(
    log_params: lmfit.Parameters,
    omega: np.ndarray,
    data: np.ndarray,
    model: lmfit.Model,
    weights: np.ndarray,
    reg_factor: float,
) -> np.ndarray:
    """
    Residual vector for a single impedance spectrum with L1 regularization.

    Parameters
    ----------
    log_params : lmfit.Parameters
        Current parameter values in log space. Transformed to linear space
        internally before evaluating the model.
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    data : ndarray, shape (N,), complex
        Measured impedance values.
    model : lmfit.Model
        lmfit Model returned by ``make_lmfit_model``.
    weights : ndarray, shape (N,), real
        Per-frequency weighting factors applied to the residuals.
        A common choice is ``1 / np.abs(data) ** 0.5``.
    reg_factor : float
        L1 regularization strength. Set to 0 to disable.

    Returns
    -------
    residuals : ndarray, shape (2*N + n_params,), real
        Concatenation of [weighted real residuals, weighted imag residuals,
        L1 penalty terms].
    """
    linear_params = to_linear(log_params)
    z_model = model.eval(linear_params, freq=omega)

    residual_real = weights * (z_model.real - data.real)
    residual_imag = weights * (z_model.imag - data.imag)

    log_values = np.array([p.value for p in log_params.values()])
    l1_penalty = reg_factor * log_values

    return np.concatenate([residual_real, residual_imag, l1_penalty])


def simultaneous_residuals(
    global_params: lmfit.Parameters,
    omega: np.ndarray,
    data_list: list,
    model: lmfit.Model,
    weights: np.ndarray,
    reg_factor: float,
    smt_factor: float,
) -> np.ndarray:
    """
    Residual vector for a simultaneous fit over multiple impedance spectra
    with L1 regularization and temporal smoothness penalties.

    All spectra share the same frequency axis and the same weighting.
    Each spectrum has its own independent set of parameters, named
    ``{param}_t{t}`` (e.g. ``a0_t0``, ``a0_t1``, ...) in global_params.

    Parameters
    ----------
    global_params : lmfit.Parameters
        All parameters for all time points, named ``{param}_t{t}``.
        Values are in log space; transformed to linear internally.
    omega : ndarray, shape (N,)
        Angular frequencies in rad/s.
    data_list : list of ndarray, each shape (N,), complex
        Measured impedance at each time point.
    model : lmfit.Model
        lmfit Model returned by ``make_lmfit_model``.
    weights : ndarray, shape (N,), real
        Per-frequency weighting factors, applied equally to all spectra.
    reg_factor : float
        L1 regularization strength applied to all parameters.
    smt_factor : float
        Temporal smoothness penalty strength applied to first- and
        second-order differences of each parameter across time.

    Returns
    -------
    residuals : ndarray, real
        Concatenation of [fitting residuals for all spectra, L1 penalty,
        first-derivative penalty, second-derivative penalty].
    """
    n_spectra = len(data_list)
    param_names = model.param_names

    # --- build parameter matrix: shape (n_params, n_spectra) ---
    # values are in log space
    param_matrix = np.array([
        [global_params[f"{name}_t{t}"].value for t in range(n_spectra)]
        for name in param_names
    ])

    # --- fitting residuals ---
    fitting_residuals = []
    for t, z_data in enumerate(data_list):
        local_log = lmfit.Parameters()
        for i, name in enumerate(param_names):
            local_log.add(name, value=param_matrix[i, t])
        local_lin = to_linear(local_log)
        z_model = model.eval(local_lin, freq=omega)
        fitting_residuals.append(weights * (z_model.real - z_data.real))
        fitting_residuals.append(weights * (z_model.imag - z_data.imag))

    # --- L1 penalty ---
    l1_penalty = reg_factor * param_matrix.ravel()

    # --- temporal smoothness penalties ---
    temporal = []
    if n_spectra > 1:
        first_diff = np.diff(param_matrix, axis=1)          # (n_params, n_spectra-1)
        temporal.append((smt_factor * first_diff).ravel())
    if n_spectra > 2:
        second_diff = (param_matrix[:, 2:]
                       - 2 * param_matrix[:, 1:-1]
                       + param_matrix[:, :-2])               # (n_params, n_spectra-2)
        temporal.append((smt_factor * 0.5 * second_diff).ravel())

    blocks = [np.concatenate(fitting_residuals), l1_penalty]
    if temporal:
        blocks.append(np.concatenate(temporal))

    return np.concatenate(blocks)
