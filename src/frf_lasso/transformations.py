"""
Log-space parameter transformation for numerical conditioning.

Rational polynomial coefficients can span many orders of magnitude and must
remain positive for the log transform to be defined (negative coefficients are
handled by fitting their absolute value and carrying the sign separately — this
is done in fitting.py).  Working in log space has two benefits:

  - The optimiser takes multiplicative steps rather than additive ones, which
    is better suited to parameters that vary over many decades.
  - The L1 penalty in objective.py is applied to the log values, which
    penalises the *order of magnitude* of the coefficients rather than their
    raw magnitude, giving a more balanced regularisation across parameters of
    very different scales.

These functions are internal utilities called by fitting.py.  They are not
part of the public API.
"""

import numpy as np
import lmfit


def to_log(params: lmfit.Parameters) -> lmfit.Parameters:
    """
    Return a copy of *params* with all values transformed to log space.

    Bounds are transformed consistently: finite bounds are log-transformed,
    infinite bounds (lmfit's defaults of -inf / +inf) are left as-is so that
    lmfit does not reject the parameters.

    Parameters
    ----------
    params : lmfit.Parameters
        Parameters with strictly positive values. Raises ValueError if any
        value is zero or negative.

    Returns
    -------
    lmfit.Parameters
        New Parameters object with the same names and log-transformed values
        and bounds.
    """
    log_params = lmfit.Parameters()

    for name, param in params.items():
        if param.value <= 0:
            raise ValueError(
                f"Parameter '{name}' has value {param.value}; log transform "
                "requires strictly positive values."
            )
        log_value = np.log(param.value)
        log_min = np.log(param.min)  if np.isfinite(param.min)  else -np.inf
        log_max = np.log(param.max)  if np.isfinite(param.max)  else  np.inf
        log_params.add(name, value=log_value, min=log_min, max=log_max)

    return log_params


def to_linear(params: lmfit.Parameters) -> lmfit.Parameters:
    """
    Return a copy of *params* with all values transformed back to linear space.

    This is the inverse of ``to_log``.  Finite bounds are exponentiated;
    infinite bounds are left as-is.

    Parameters
    ----------
    params : lmfit.Parameters
        Parameters in log space (i.e. output of ``to_log``).

    Returns
    -------
    lmfit.Parameters
        New Parameters object with exponentiated values and bounds.
    """
    linear_params = lmfit.Parameters()

    for name, param in params.items():
        linear_value = np.exp(param.value)
        linear_min = np.exp(param.min) if np.isfinite(param.min) else -np.inf
        linear_max = np.exp(param.max) if np.isfinite(param.max) else  np.inf
        linear_params.add(name, value=linear_value, min=linear_min, max=linear_max)

    return linear_params
