import json
from typing import Tuple, List
import numpy as np
from lmfit.model import Model, ModelResult, load_model, load_modelresult
from lmfit import Parameters, Model 
from lmfit.minimizer import MinimizerResult

def dict_to_result(result_dict, model):
    """Convert dictionary back to MinimizerResult"""
    # Create parameters
    params = Parameters()
    for name, param_data in result_dict['params'].items():
        params.add(name, value=param_data['value'], vary=param_data['vary'],
                  min=param_data['min'], max=param_data['max'], expr=param_data['expr'])
        if param_data['stderr'] is not None:
            params[name].stderr = param_data['stderr']
    
    # Create a minimal result object
    result = MinimizerResult()
    result.method = result_dict['method']
    result.params = params
    result.success = result_dict['success']
    result.errorbars = result_dict['errorbars']
    result.message = result_dict['message']
    result.nfev = result_dict['nfev']
    result.chisqr = result_dict['chisqr']
    result.redchi = result_dict['redchi']
    result.aic = result_dict['aic']
    result.bic = result_dict['bic']
    result.ndata = result_dict['ndata']
    result.nvarys = result_dict['nvarys']
    result.nfree = result_dict['nfree']
    result.residual = np.array(result_dict['residual']) if result_dict['residual'] is not None else None
    if result_dict['flatchain'] is not None:
        result.flatchain = np.array(result_dict['flatchain'])

    return result

def load_simultaneous(path)-> Tuple[Model, MinimizerResult,  List[np.ndarray],List[np.ndarray]]:
    # Load model
    model = load_model(path + '/rational_poly_model.sav')
    # Load result
    with open(path + '/result.json') as file:  
        result_dict = json.load(file) 
    result = dict_to_result(result_dict, model)
    # Load data
    impedance_array = np.load(path+"/impedance_set.npy", allow_pickle=True)
    impedance_set = [np.array(obj, dtype=np.complex128) for obj in impedance_array]
    # Load fitting
    fits_array = np.load(path+"/fits.npy", allow_pickle=True)
    fits = [np.array(obj, dtype=np.complex128) for obj in fits_array]
    return  model, result, impedance_set, fits

def load_sequential(path) -> Tuple[Model, np.ndarray, List[MinimizerResult], List[np.ndarray],List[np.ndarray]]:
    # Load model
    model = load_model(path + '/rational_poly_model.sav')
    # Load weighting factors
    weighting_factors = np.load(path+"/weighting_factors.npy")
    # Load results list
    with open(path + '/results.json', 'r') as f:  
        results_dicts = json.load(f)  
    results = [dict_to_result(result_dict, model) for result_dict in results_dicts]
    # Load data
    impedance_array = np.load(path+"/impedance_set.npy", allow_pickle=True)
    impedance_set = [np.array(obj, dtype=np.complex128) for obj in impedance_array]
    # Load fitting
    fits_array = np.load(path+"/fits.npy", allow_pickle=True)
    fits = [np.array(obj, dtype=np.complex128) for obj in fits_array]
    # Load frequencies
    frequencies = np.load(path+"/frequencies.npy")
    return model, weighting_factors, results, impedance_set, fits, frequencies

def load_multistart(path) -> Tuple[Model, List[MinimizerResult], np.ndarray, np.ndarray]:
    # Load model
    model = load_model(path + '/rational_poly_model.sav')
    # Load results list
    with open(path + '/results.json', 'r') as f:  
        results_dicts = json.load(f)  
    results = [dict_to_result(result_dict, model) for result_dict in results_dicts]
    # Load data
    impedance = np.loadtxt(path+"/impedance.txt", dtype = np.complex64)
    # Load fitting
    fits_array = np.load(path+"/fits.npy", allow_pickle=True)
    fits = [np.array(obj, dtype=np.complex128) for obj in fits_array]
    return model, results, impedance, fits

def load(path) -> Tuple[Model, MinimizerResult, np.ndarray, np.ndarray]:
    # Load model
    model = load_model(path + '/rational_poly_model.sav')
    # Load result
    with open(path + '/result.json') as file:  
        result_dict = json.load(file) 
    result = dict_to_result(result_dict, model)
    # Load data
    impedance = np.loadtxt(path+"/impedance.txt", dtype = np.complex64)
    # Load fitting
    fit = np.loadtxt(path+"/fit.txt", dtype = np.complex64)
    return model, result, impedance, fit