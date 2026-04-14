"""
frf_lasso — rational polynomial fitting for electrochemical impedance spectra.

Top-level imports
-----------------
The core fitting functions and model factory are available directly:

    from frf_lasso import fit_single, fit_multistart, fit_sequential
    from frf_lasso import fit_simultaneous, adapt_params
    from frf_lasso import make_lmfit_model

I/O, statistics, and visualization are in dedicated submodules:

    from frf_lasso.io import save_single, load_single
    from frf_lasso.statistics import compare_fits, compare_sequential_fits
    from frf_lasso.visualization import nyquist_plot, slider_plot
"""

from .models import make_lmfit_model, rational_poly
from .fitting import fit_single, fit_multistart, fit_sequential
from .simultaneous import fit_simultaneous, adapt_params

__version__ = "0.1.0"

__all__ = [
    # models
    "make_lmfit_model",
    "rational_poly",
    # fitting
    "fit_single",
    "fit_multistart",
    "fit_sequential",
    # simultaneous
    "fit_simultaneous",
    "adapt_params",
]
