import lmfit
import numpy as np
from math import log10
from transformations import transform_params_to_log, transform_params_to_linear
    
def rational_poly_fit(frequencies, impedance, model_fun, weights, params, objective_fun, obj_args=None, method = 'least_squares'):
    # Normalize imput data
    # impedance_scaled = impedance/np.max(np.abs(impedance))
    params = transform_params_to_log(params)
    result = lmfit.minimize(
        objective_fun, 
        params,
        args=(frequencies, impedance, model_fun, weights, obj_args), 
        method = 'least_squares',
        max_nfev=50000,
    )
    result.params = transform_params_to_linear(result.params)
    fit = model_fun.eval(result.params, x=frequencies*2*np.pi)
    # Rescale the fit
    # fit = fit * np.max(np.abs(impedance))
    return result, fit


def multiple_start_rational_poly_fit(frequencies, impedance, model_fun, objective_fun, obj_args=None, n_trials=10, seed=42, min=1e-6, max = 1e6):
    # Normalize imput data
    # impedance_scaled = impedance/np.max(np.abs(impedance))
    np.random.seed(seed)
    results = []
    fits = []
    for trial in range(n_trials):
        # Create a set of Parameters
        params = lmfit.Parameters()
        for p in model_fun.param_names:
            params.add(p, value= 0.5, min =min, max = max)#value = np.random.rand(1) * max, min=min, max = max)
        # Minimize
        global_result, fit = rational_poly_fit(
            frequencies,
            impedance,
            model_fun,
            params,
            objective_fun,
            obj_args,
            method = 'basinhopping'
        )
        result, fit = rational_poly_fit(
            frequencies,
            impedance,
            model_fun,
            global_result.params,
            objective_fun,
            obj_args,
        )
        results.append(result)
        fits.append(fit)
    return results, fits


def sequential_rational_poly_fit(frequencies, impedance_set, model_fun, weights_list, params, objective_fun, obj_args=None, method = 'least_square'):
    results = []
    fits = []
    for i, impedance in enumerate(impedance_set):
        # Set initial parameters
        if i == 0:
            # Use your original params for first fit
            current_params = params.copy()
            current_params = transform_params_to_log(current_params)
        else:
            # Use previous result as starting point
            current_params = results[i-1].params.copy()
            current_params = transform_params_to_log(current_params)
        # Get previous parameters for temporal regularization
        # if i > 0: 
        #     previous_params = results[i-1].params  
        #     # previous_params = transform_params_to_log(previous_params)
        # else: 
        #     previous_params = None
        # Normalize input data
        # impedance_scaled = impedance/np.max(np.abs(impedance))
        result = lmfit.minimize(
            objective_fun, 
            current_params,
            args=(frequencies, impedance, model_fun, weights_list[i], obj_args), 
            method = method,
            max_nfev=50000,
        )
        result.params = transform_params_to_linear(result.params)
        fit = model_fun.eval(result.params, x=frequencies*2*np.pi)
        # Rescale the fit
        # fit = fit * np.max(np.abs(impedance))
        results.append(result)
        fits.append(fit)
    return results, fits