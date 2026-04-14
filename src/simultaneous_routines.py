import lmfit, time
from lmfit.parameter import Parameters
from typing import List
import numpy as np
from obj_functions import extract_params_for_time, complex_least_square_regl1, simultanous_least_square_regl1_tsmooth
from routines import rational_poly_fit, sequential_rational_poly_fit
from data_savers import save_simultaneous, save_sequential
from fit_statistics import  extract_global_param_evolution
from transformations import transform_params_to_log, transform_params_to_linear
from lmfit.model import ModelResult, Model

def create_global_parameters_from_sequential(sequential_results: List[ModelResult], model_fun:Model):
    """Create global parameters initialized with sequential results"""
    print("\nInitializing global parameters from sequential results...")
    
    global_params = lmfit.Parameters()
    param_names = model_fun.param_names
    
    for t, result in enumerate(sequential_results):
        for param_name in param_names:
            param_key = f'{param_name}_t{t}'
            initial_value = result.params[param_name].value
            
            # Use same bounds as original
            global_params.add(param_key, value=initial_value, min=1e-6, max=10000)
    
    print(f"Created {len(global_params)} global parameters")
    return global_params
# def create_global_parameters_from_sequential(sequential_results, model_fun):
#     """Optimized version using batch operations"""
#     print("\nInitializing global parameters from sequential results...")
    
#     global_params = lmfit.Parameters()
#     param_names = model_fun.param_names
#     n_spectra = len(sequential_results)
    
#     # Pre-allocate parameter values array for better memory access patterns
#     param_values = np.zeros((n_spectra, len(param_names)))
    
#     # Extract all parameter values in one pass
#     for t, result in enumerate(sequential_results):
#         for i, param_name in enumerate(param_names):
#             param_values[t, i] = result.params[param_name].value
    
#     # Create parameters in batch
#     for t in range(n_spectra):
#         for i, param_name in enumerate(param_names):
#             param_key = f'{param_name}_t{t}'
#             global_params.add(param_key, value=param_values[t, i], min=1e-6, max=10000)
    
#     print(f"Created {len(global_params)} global parameters")
#     return global_params

def global_fitting_with_temporal_smoothing(frequencies, impedance_set, model_fun, 
                                         params, objective_fun, obj_args=None, method = 'lest_squares'):
    start_time = time.time()
    # Create global parameters from sequential results
    # global_params = create_global_parameters_from_sequential(params, model_fun)
    global_params = params
    # global_params = transform_params_to_log(global_params) 
    print("Starting global optimization...")
    print(f"Penalty weights: {obj_args['smt_factor']}")
    # Global optimization
    result = lmfit.minimize(
        objective_fun,
        global_params,
        args=(frequencies, impedance_set, model_fun, obj_args),
        method=method,
        max_nfev=50000,
    )
    # result.params = transform_params_to_linear(result.params)
    global_time = time.time() - start_time
    print(f"Global fitting completed in {global_time:.2f} seconds")
    print(f"Final χ² = {result.chisqr:.3e}")
    # Extract individual fits
    fits = []
    individual_params = []
    for t in range(len(impedance_set)):
        local_params = extract_params_for_time(result.params, model_fun.param_names, t)
        fit = model_fun.eval(local_params, x = frequencies*2*np.pi)
        # Rescale
        # fit = fit * np.max(np.abs(impedance_set[t]))
        fits.append(fit)
        individual_params.append(local_params)
    return result, fits, individual_params, global_time

def simultaneous_fitting(
    model,
    impedance, 
    frequencies,
    base_saving_path,
    reg_factor=1e-4,
    smt_factor=1e-3,
    initial_params=None  # Add this parameter
):
    # Process input data
    impedance_set = [impedance[:,i] for i in range(impedance.shape[1])]
    if initial_params is not None:
        # Use parameters from previous cycle
        params = initial_params.copy()
    else:
        # Create random initial parameters (for first cycle)
        params = lmfit.Parameters()
        np.random.seed(42)
        for key in model.param_names:
            params[key] = lmfit.Parameter(name=key, value=np.random.lognormal(-6, 6), min=1e-6, max = 1e6)
    # Find initial parameters from one spectra
    obj_args={
        'reg_factor': reg_factor,
    }
    middel_index = len(impedance_set)//2
    global_result, global_fit = rational_poly_fit(
        frequencies,
        impedance_set[middel_index],
        model,
        params,
        complex_least_square_regl1,
        obj_args,
        method='differential_evolution'
    )
    # Refine the fitting
    result, fit = rational_poly_fit(
        frequencies,
        impedance_set[middel_index],
        model,
        global_result.params,
        complex_least_square_regl1,
        obj_args,
    )
    # Fit
    seq_results, seq_fits = sequential_rational_poly_fit(
        frequencies,
        impedance_set,
        model,
        result.params,
        complex_least_square_regl1,
        obj_args,
    )
    saving_path = base_saving_path +"/sequential"
    save_sequential(
        saving_path,
        model,
        seq_results,
        impedance_set,
        seq_fits,
    )
    global_params = create_global_parameters_from_sequential(seq_results, model)
    final_result, fits, individual_params, global_time = global_fitting_with_temporal_smoothing(
        frequencies, 
        impedance_set, 
        model, 
        global_params, 
        simultanous_least_square_regl1_tsmooth, 
        {
            "reg_factor": obj_args['reg_factor'],
            "smt_factor": smt_factor,
        },
        )
    # Save results
    saving_path = base_saving_path
    save_simultaneous(
        saving_path,
        model,
        final_result,
        impedance_set,
        fits,
    )
   
    return final_result.params

def simultaneous_fitting_previous_params(
    model,
    impedance, 
    frequencies,
    base_saving_path,
    initial_params,
    reg_factor=1e-4,
    smt_factor=1e-3,
):
    # Process input data
    impedance_set = [impedance[:,i] for i in range(impedance.shape[1])]
    final_result, fits, individual_params, global_time = global_fitting_with_temporal_smoothing(
        frequencies, 
        impedance_set, 
        model, 
        initial_params, 
        simultanous_least_square_regl1_tsmooth, 
        {
            "reg_factor": reg_factor,
            "smt_factor": smt_factor,
        },
        )
    # Save results
    saving_path = base_saving_path
    save_simultaneous(
        saving_path,
        model,
        final_result,
        impedance_set,
        fits,
    )
    return final_result.params

def adapt_params(params_names, previous_params, impedance):
    """
    Adapt parameters when the number of spectra changes between iterations.
    
    Args:
        params_names: List of parameter names from the model
        previous_params: lmfit.Parameters object from previous iteration
        impedance: Current impedance data array
    
    Returns:
        new_params: Adapted lmfit.Parameters object
    """
    previous_spectra_num = len(previous_params) // len(params_names)  # Integer division
    current_spectra_num = impedance.shape[1]
    previous_params_names = list(previous_params.keys())
    len_diff = current_spectra_num - previous_spectra_num
    
    if len_diff > 0:
        # More spectra than before - need to add parameters
        print(f"Adding parameters for {len_diff} additional spectra")
        
        # Create a copy of previous parameters
        new_params = previous_params.copy()
        
        # Find the parameters from the last time point to use as template
        last_time_idx = previous_spectra_num - 1
        
        # Add parameters for new time points
        for new_time_idx in range(previous_spectra_num, current_spectra_num):
            for param_name in params_names:
                # Use the last spectrum's parameters with EXACT same values
                template_param_key = f'{param_name}_t{last_time_idx}'
                new_param_key = f'{param_name}_t{new_time_idx}'
                
                if template_param_key in previous_params:
                    # Copy the exact value from the last time point
                    template_value = previous_params[template_param_key].value
                    new_params.add(new_param_key, value=template_value, min=1e-12, max=1e12)
                else:
                    # Fallback if template not found
                    print(f"Warning: Template parameter {template_param_key} not found, using random initialization")
                    new_params.add(new_param_key, value=np.random.lognormal(-12, 12), min=1e-12, max=1e12)
                    
    elif len_diff < 0:
        # Fewer spectra than before - need to remove parameters
        print(f"Removing parameters for {-len_diff} spectra")
        new_params = previous_params.copy()
        
        # Remove parameters from the end (highest time indices)
        params_to_remove = []
        for time_idx in range(current_spectra_num, previous_spectra_num):
            for param_name in params_names:
                param_key = f'{param_name}_t{time_idx}'
                if param_key in new_params:
                    params_to_remove.append(param_key)
        
        # Remove the parameters
        for param_key in params_to_remove:
            del new_params[param_key]
            
    else:
        # Same number of spectra - no change needed
        new_params = previous_params.copy()
    
    print(f"Parameter adaptation: {previous_spectra_num} -> {current_spectra_num} spectra")
    print(f"Total parameters: {len(previous_params)} -> {len(new_params)}")
    
    return new_params