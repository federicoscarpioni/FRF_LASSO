import numpy as np
import lmfit

def transform_params_to_log(linear_params: lmfit.Parameters) -> lmfit.Parameters:
    """Transform parameter values to log space, keeping same names"""
    log_params = lmfit.Parameters()
    
    for name, param in linear_params.items():
        log_value = np.log(param.value)
        log_min = np.log(param.min)
        log_max = np.log(param.max)
        log_params.add(name, value=log_value, min=log_min, max=log_max)
    
    return log_params

def transform_params_to_linear(log_params: lmfit.Parameters) -> lmfit.Parameters:
    """Transform parameter values back to linear space, keeping same names"""
    linear_params = lmfit.Parameters()
    
    for name, param in log_params.items():
        linear_value = np.exp(param.value)
        # You might want to use original bounds here, or transform them back
        linear_min = np.exp(param.min)
        linear_max = np.exp(param.max)
        
        linear_params.add(name, value=linear_value, min=linear_min, max=linear_max)
    
    return linear_params

def log_space_regularization(log_params: lmfit.Parameters, reg_factor:float) -> float:
    log_values = np.array([param.value for param in log_params.values()])
    return reg_factor * log_values