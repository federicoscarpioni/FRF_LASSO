import numpy as np
import lmfit
from transformations import transform_params_to_linear

def complex_least_square(params, frequencies, data, model_fun, weights, previous_params=None, obj_args = None):
    # Constract complex least square
    params_lin = transform_params_to_linear(params)
    model = model_fun.eval(params_lin, x = frequencies*2*np.pi)
    residual_real = weights * (model.real - data.real)
    residual_imag = weights * (model.imag - data.imag)
    return np.concatenate([residual_real, residual_imag])

def complex_least_square_regl1(params, frequencies, data, model_fun, weights, previous_params=None, obj_args={'reg_factor':1e-3}):
    '''
    obj_args must be a dictionary that contains 'reg_factor'
    '''
    # Define a weighting factor
    # wf = (frequencies)**(0.2)#1/(np.abs(data))**2
    # Constract complex least square
    params_lin = transform_params_to_linear(params)
    model = model_fun.eval(params_lin, x = frequencies*2*np.pi)
    residual_real = weights * (model.real - data.real)
    residual_imag = weights * (model.imag - data.imag)
    # L1 regularization terms 
    param_values = np.array([params_lin[name].value for name in params_lin.keys()])
    regularization = obj_args['reg_factor'] * param_values
    return np.concatenate([residual_real, residual_imag, regularization])

def complex_least_square_regl1_tsmooth(global_params, frequencies, data_list, model_fun, previous_params=None, obj_args=None):

    n_spectra = len(data_list)
    param_names = model_fun.param_names
    all_residuals = []
    
    # STEP 1: Calculate residual of all spectra
    param_matrix = np.array([
        [global_params[f'{param_name}_t{t}'].value for t in range(n_spectra)]
        for param_name in param_names
    ])  # Shape: [n_params, n_time_points]
    
    # 1. Fitting residuals (this loop is harder to vectorize due to different scaling per spectrum)
    for t, impedance_data in enumerate(data_list):
        # impedance_scaled = impedance_data / np.max(np.abs(impedance_data))
        
        local_params = lmfit.Parameters()
        for i, param_name in enumerate(param_names):
            local_params.add(param_name, value=param_matrix[i, t])
        
        # local_params = transform_params_to_linear(local_params)
        model = model_fun.eval(local_params, x = frequencies * 2 * np.pi)
        
        wf = 1#1/np.abs(impedance_data)
        residual_real = wf * (model.real - impedance_data.real)
        residual_imag = wf * (model.imag - impedance_data.imag)
        all_residuals.extend(residual_real)
        all_residuals.extend(residual_imag)
    
    # STEP 2: Regularization in one operation
    all_param_values = param_matrix.flatten()  # All parameters as 1D array
    # all_param_values=transform_params_to_linear(all_param_values)
    regularization = obj_args['reg_factor'] * all_param_values
    all_residuals.extend(regularization)
    
    # STEP 3: Temporal penalties with pure NumPy
    weight = obj_args['smt_factor']
    temporal_penalties = []
    
    if n_spectra > 1:
        # First derivatives for all parameters at once
        first_derivs = np.diff(param_matrix, axis=1)  # Shape: [n_params, n_time-1]
        temporal_penalties.extend((weight * first_derivs).flatten())
    
    if n_spectra > 2:
        # Second derivatives for all parameters at once
        second_derivs = param_matrix[:, 2:] - 2*param_matrix[:, 1:-1] + param_matrix[:, :-2]
        temporal_penalties.extend((weight * 0.5 * second_derivs).flatten())
    
    all_residuals.extend(temporal_penalties)
    
    return np.array(all_residuals)

def extract_params_for_time(global_params, param_names, time_index):
    """Extract parameters for a specific time point"""
    local_params = lmfit.Parameters()
    for param_name in param_names:
        param_key = f'{param_name}_t{time_index}'
        local_params.add(param_name, value=global_params[param_key].value)
    return local_params

def simultanous_least_square_regl1_tsmooth(global_params, frequencies, data_list, model_fun, obj_args=None):
    n_spectra = len(data_list)
    param_names = model_fun.param_names
    all_residuals = []
    # 1. Fitting residuals for all spectra
    for t, impedance in enumerate(data_list):
        # Scale impedance
        # impedance_scaled = impedance / np.max(np.abs(impedance))
        # Extract parameters for this time point
        local_params = extract_params_for_time(global_params, param_names, t)
        # Calculate model
        model = model_fun.eval(local_params, x = frequencies * 2 * np.pi)
        # Weighting factor
        wf = (frequencies)**1 #1 / np.abs(impedance)
        # Complex residuals
        residual_real = wf * (model.real - impedance.real)
        residual_imag = wf * (model.imag - impedance.imag)
        all_residuals.extend(residual_real)
        all_residuals.extend(residual_imag)
    # 2. Parameter regularization
    regularization = []
    for param_key in global_params.keys():
        regularization.append(obj_args['reg_factor'] * global_params[param_key].value)
    all_residuals.extend(regularization)
    
    # 3. Temporal smoothness penalties
    temporal_penalties = []
    
    for param_name in param_names:
        param_series = []
        for t in range(n_spectra):
            param_key = f'{param_name}_t{t}'
            param_series.append(global_params[param_key].value)
        
        param_series = np.array(param_series)
        # !!! This implementation leaves open the option to set a differenet smoothing factor per parameter
        weight = obj_args['smt_factor'] 
        # First derivative penalty (smoothness)
        if n_spectra > 1:
            first_deriv = np.diff(param_series)
            temporal_penalties.extend(weight * first_deriv)
        # Second derivative penalty (avoid sharp changes)
        if n_spectra > 2:
            second_deriv = param_series[2:] - 2*param_series[1:-1] + param_series[:-2]
            temporal_penalties.extend(weight * 0.5 * second_deriv)
    
    all_residuals.extend(temporal_penalties)
    
    return np.array(all_residuals)

# def simultanous_least_square_regl1_tsmooth(global_params, frequencies, data_list, model_fun, obj_args=None):
#     n_spectra = len(data_list)
#     n_freq = len(frequencies)
#     param_names = model_fun.param_names
#     n_params = len(param_names)
    
#     # Pre-calculate sizes for efficient array allocation
#     n_fitting = 2 * n_freq * n_spectra  # real + imag for all spectra
#     n_reg = len(global_params)
#     n_temporal = n_params * (2 * n_spectra - 3) if n_spectra > 2 else n_params * (n_spectra - 1) if n_spectra > 1 else 0
    
#     # Pre-allocate result array
#     total_residuals = np.zeros(n_fitting + n_reg + n_temporal)
    
#     # Convert global_params to arrays for faster access
#     param_values = np.array([global_params[key].value for key in global_params.keys()])
#     param_keys = list(global_params.keys())
    
#     # Create parameter lookup for faster access
#     param_dict = {key: i for i, key in enumerate(param_keys)}
    
#     # 1. Fitting residuals - vectorized where possible
#     residual_idx = 0
#     omega = frequencies * 2 * np.pi
    
#     for t, impedance in enumerate(data_list):
#         # Fast parameter extraction using pre-computed indices
#         local_param_values = np.zeros(n_params)
#         for i, param_name in enumerate(param_names):
#             param_key = f'{param_name}_t{t}'
#             local_param_values[i] = param_values[param_dict[param_key]]
        
#         # Create local params dict more efficiently
#         local_params = {param_names[i]: local_param_values[i] for i in range(n_params)}
        
#         # Calculate model
#         model = model_fun.eval(local_params, x=omega)
        
#         # Complex residuals - direct array assignment
#         residual_real = model.real - impedance.real
#         residual_imag = model.imag - impedance.imag
        
#         # Assign to pre-allocated array
#         total_residuals[residual_idx:residual_idx + n_freq] = residual_real
#         total_residuals[residual_idx + n_freq:residual_idx + 2*n_freq] = residual_imag
#         residual_idx += 2 * n_freq
    
#     # 2. Parameter regularization - vectorized
#     reg_start = residual_idx
#     total_residuals[reg_start:reg_start + n_reg] = obj_args['reg_factor'] * param_values
    
#     # 3. Temporal smoothness penalties - vectorized
#     temporal_start = reg_start + n_reg
#     temporal_idx = temporal_start
    
#     if n_spectra > 1:
#         # Reshape parameter values for efficient temporal operations
#         param_matrix = param_values.reshape(n_spectra, n_params)  # Assuming params are ordered by time
        
#         weight = obj_args['smt_factor']
        
#         # First derivative penalty
#         first_deriv = np.diff(param_matrix, axis=0).flatten()
#         n_first = len(first_deriv)
#         total_residuals[temporal_idx:temporal_idx + n_first] = weight * first_deriv
#         temporal_idx += n_first
        
#         # Second derivative penalty
#         if n_spectra > 2:
#             second_deriv = (param_matrix[2:] - 2*param_matrix[1:-1] + param_matrix[:-2]).flatten()
#             n_second = len(second_deriv)
#             total_residuals[temporal_idx:temporal_idx + n_second] = weight * 0.5 * second_deriv
    
#     return total_residuals