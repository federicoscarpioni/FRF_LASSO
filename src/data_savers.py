import json, lmfit, os
import numpy as np

def result_to_dict(result):
    """Convert MinimizerResult to dictionary"""
    result_dict = {
        'method': result.method,
        'params': {name: {'value': param.value, 'vary': param.vary, 
                         'min': param.min, 'max': param.max, 'expr': param.expr,
                         'stderr': param.stderr} for name, param in result.params.items()},
        'success': result.success,
        'errorbars': result.errorbars,
        'message': result.message,
        'nfev': result.nfev,
        'chisqr': result.chisqr,
        'redchi': result.redchi,
        'aic': result.aic,
        'bic': result.bic,
        'ndata': result.ndata,
        'nvarys': result.nvarys,
        'nfree': result.nfree,
        'residual': result.residual.tolist() if result.residual is not None else None,
        'flatchain': result.flatchain.tolist() if hasattr(result, 'flatchain') and result.flatchain is not None else None
    }
    return result_dict

def save_simultaneous(path, model, result, impedance_set,fits):
    # Ensure path exists
    os.makedirs(path, exist_ok=True)
    # Save lmfit model
    lmfit.model.save_model(model, path+'/rational_poly_model.sav')
    # Save lmfit fitting result
    results_dicts = result_to_dict(result)
    with open(path+'/result.json', 'w') as file:  
        json.dump(results_dicts, file, indent=2)
    # Save data
    impedance_array = np.array(impedance_set, dtype=object)
    np.save(path+"/impedance_set.npy", impedance_array)
    # Save fitting
    fits_array = np.array(fits, dtype=object)
    np.save(path+"/fits.npy", fits_array)


def save_sequential(path, model, weights, results,impedance_set,fits,frequencies):
    # Ensure path exists
    os.makedirs(path, exist_ok=True)
    # Save lmfit model
    lmfit.model.save_model(model, path+'/rational_poly_model.sav')
    # Save wighting factors
    np.save(path+"/weighting_factors.npy", weights)
    # Save lmfit fitting results
    results_dicts = [result_to_dict(result) for result in results]
    with open(path+'/results.json', 'w') as file:  
        json.dump(results_dicts, file, indent=2)
    # Save data
    impedance_array = np.array(impedance_set, dtype=object)
    np.save(path+"/impedance_set.npy", impedance_array)
    # Save fitting
    fits_array = np.array(fits, dtype=object)
    np.save(path+"/fits.npy", fits_array)
    # Save frequencies
    np.save(path+"/frequencies.npy", frequencies)


def save_multistart(path, model, results, impedance, fits):
    # Ensure path exists
    os.makedirs(path, exist_ok=True)
    # Save lmfit model
    lmfit.model.save_model(model, path+'/rational_poly_model.sav')
    # Save lmfit fitting results
    results_dicts = [result_to_dict(result) for result in results]
    with open(path+'/results.json', 'w') as file:  
        json.dump(results_dicts, file, indent=2)
    # Save data
    np.savetxt(path+"/impedance.txt", impedance)
    # Save fitting
    fits_array = np.array(fits, dtype=object)
    np.save(path+"/fits.npy", fits_array)


def save(path, model, result, impedance, fit):
    # Ensure path exists
    os.makedirs(path, exist_ok=True)
    # Save lmfit model
    lmfit.model.save_model(model, path+'/rational_poly_model.sav')
    # Save lmfit fitting result
    results_dicts = result_to_dict(result)
    with open(path+'/result.json', 'w') as file:  
        json.dump(results_dicts, file, indent=2)
    # Save data
    np.savetxt(path+"/impedance.txt", impedance)
    # Save fitting
    np.savetxt(path+"/fit.txt", fit)