"""
Rational polynomial model for electrochemical impedance spectra.

Design notes
------------
**Separation of pure function and lmfit object**
    ``rational_poly`` is a plain NumPy function with no dependency on lmfit.
    It can be called directly to evaluate the model — useful when loading saved
    coefficients from disk, computing a prediction after fitting, or writing
    tests.  ``make_lmfit_model`` is the only place that knows about lmfit: it
    wraps ``rational_poly`` in a closure whose keyword-argument signature
    satisfies lmfit's parameter-introspection mechanism.  Keeping the two
    concerns separate means the core math can be understood and tested on its
    own, and the lmfit integration is confined to one small function.

**Vectorised evaluation**
    Rather than expanding the polynomial term by term (``a0 + a1*s + a2*s**2
    + ...``), the implementation builds a powers matrix
    ``S = s[:, None] ** np.arange(n)`` of shape ``(N_freq, n)`` and evaluates
    both numerator and denominator with a single matrix–vector product
    ``S @ coeffs``.  This removes the explicit dependence on polynomial order
    from the computation, makes the code order-agnostic, and is marginally
    faster for large frequency arrays.
"""

import numpy as np
import lmfit


def rational_poly(freq, a_coeffs, b_coeffs):
    """
    Evaluate a rational polynomial impedance model at given frequencies.

    The model is defined as:

        Z(freq) = P(s) / Q(s)

    where  s = sqrt(i * freq)  and:

        P(s) = a0 + a1*s + a2*s^2 + ... + an*s^n
        Q(s) = 1  + b1*s + b2*s^2 + ... + bm*s^m

    The leading coefficient of the denominator is fixed to 1 so that the
    model is identifiable (otherwise numerator and denominator can scale
    arbitrarily while their ratio stays constant).

    Numerator and denominator can in principle have different orders n and m
    (len(a_coeffs) - 1 and len(b_coeffs) respectively). For electrochemical
    systems n = m is the standard choice and is the default in
    ``make_lmfit_model``.

    Parameters
    ----------
    freq : array_like, shape (N,)
        Frequency values. Must use the same unit (Hz or rad/s) consistently
        across fitting and evaluation — the coefficients absorb the scaling.
    a_coeffs : array_like, shape (n + 1,)
        Numerator coefficients [a0, a1, ..., an].
    b_coeffs : array_like, shape (m,)
        Denominator coefficients [b1, b2, ..., bm].

    Returns
    -------
    Z : ndarray, shape (N,), complex
        Complex impedance at each frequency point.
    """
    freq = np.asarray(freq, dtype=float)
    a = np.asarray(a_coeffs, dtype=complex)
    b = np.asarray(b_coeffs, dtype=complex)

    s = np.sqrt(1j * freq)                               # shape (N,)

    # Numerator: sum over k=0..n of a_k * s^k
    a_powers = s[:, None] ** np.arange(len(a))          # shape (N, n+1)
    numerator = a_powers @ a

    # Denominator: 1 + sum over k=1..m of b_k * s^k
    b_powers = s[:, None] ** np.arange(1, len(b) + 1)  # shape (N, m)
    denominator = 1.0 + b_powers @ b

    return numerator / denominator


def make_lmfit_model(num_order, den_order=None):
    """
    Create an lmfit Model for a rational polynomial of given order(s).

    The returned model has parameters named:
        a0, a1, ..., a{num_order}   (numerator coefficients)
        b1, b2, ..., b{den_order}   (denominator coefficients)

    These names are compatible with lmfit's Parameters interface:
    use ``result.params['a0'].value`` etc. after fitting.

    Parameters
    ----------
    num_order : int
        Numerator polynomial order (must be >= 1).
    den_order : int or None, optional
        Denominator polynomial order (must be >= 1). Defaults to ``num_order``
        when None, which is the standard choice for electrochemical systems.

    Returns
    -------
    model : lmfit.Model
        lmfit Model object ready to have parameters initialised and be fitted.
        The independent variable is named 'freq'.

    Examples
    --------
    >>> model = make_lmfit_model(num_order=6)
    >>> params = model.make_params()
    >>> result = model.fit(impedance, params, freq=frequencies)

    >>> # Asymmetric case: order-6 numerator, order-4 denominator
    >>> model = make_lmfit_model(num_order=6, den_order=4)
    """
    if den_order is None:
        den_order = num_order

    if num_order < 1 or den_order < 1:
        raise ValueError(
            f"num_order and den_order must be >= 1, "
            f"got num_order={num_order}, den_order={den_order}"
        )

    a_names = [f"a{i}" for i in range(num_order + 1)]      # a0 … a_num_order
    b_names = [f"b{i}" for i in range(1, den_order + 1)]   # b1 … b_den_order
    param_names = a_names + b_names

    def _model_func(freq, **kwargs):
        a = np.array([kwargs[n] for n in a_names])
        b = np.array([kwargs[n] for n in b_names])
        return rational_poly(freq, a, b)

    if num_order == den_order:
        _model_func.__name__ = f"rational_poly{num_order}"
    else:
        _model_func.__name__ = f"rational_poly_n{num_order}_m{den_order}"

    return lmfit.Model(
        _model_func,
        independent_vars=["freq"],
        param_names=param_names,
    )
