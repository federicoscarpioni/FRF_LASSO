import json, lmfit
import numpy as np
import matplotlib.pyplot as plt
from routines import rational_poly_fit
from rational_poly_models import rational_poly6 as rational_poly
from obj_functions import complex_least_square_regl1
from routines import rational_poly_fit
from data_savers import save

# Import one spectra to fit
folder = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_010/2507281144PanCoin2025_010_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz/pico_aquisition/cycle_20_sequence_4'
impedance = np.load(folder+'/impedance_total.npy')[1:-3,100]
metadata_file = open("C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_010/2507281144PanCoin2025_010_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz/metadata_deis_exp.json")
frequencies = np.array( json.load(metadata_file)['Frequencies multisine (Hz)'] )[1:-3]

regf = 1e-12
fit_type = 'impedance'
# fit_type = 'admittance'

# Create lmfit model
model = lmfit.Model(rational_poly)

# Create a set of Parameters
params = lmfit.Parameters()
starting_value = 1e6
params.add('a0', value=starting_value, min=1e-12, max = 1e12)
params.add('a1', value=starting_value, min=1e-12, max = 1e12)
params.add('a2', value=starting_value, min=1e-12, max = 1e12)
params.add('a3', value=starting_value, min=1e-12, max = 1e12)
params.add('a4', value=starting_value, min=1e-12, max = 1e12)
params.add('a5', value=starting_value, min=1e-12, max = 1e12)
params.add('a6', value=starting_value, min=1e-12, max = 1e12)
params.add('a7', value=starting_value, min=1e-12, max = 1e12)
params.add('b1', value=starting_value, min=1e-12, max = 1e12)
params.add('b2', value=starting_value, min=1e-12, max = 1e12)
params.add('b3', value=starting_value, min=1e-12, max = 1e12)
params.add('b4', value=starting_value, min=1e-12, max = 1e12)
params.add('b5', value=starting_value, min=1e-12, max = 1e12)
params.add('b6', value=starting_value, min=1e-12, max = 1e12)
params.add('b7', value=starting_value, min=1e-12, max = 1e12)


if fit_type == 'impedance':
    # Fit impedance
    result, fit = rational_poly_fit(
        frequencies,
        impedance,
        model,
        params,
        complex_least_square_regl1,
        {'reg_factor' : regf},
        method = 'basinhopping'
    )
    result, fit = rational_poly_fit(
        frequencies,
        impedance,
        model,
        result.params,
        complex_least_square_regl1,
        {'reg_factor' : regf},
    )
elif fit_type == 'admittance':
    # Fit admittance
    result, fit = rational_poly_fit(
        frequencies,
        1/impedance,
        model,
        params,
        complex_least_square_regl1,
        {'reg_factor' : regf},
    )
    fit = 1/fit

# Save results
saving_path = './data/results/2510061349_single_spectra_test'
save(
    saving_path,
    model,
    result,
    impedance,
    fit
)

# Print report
lmfit.report_fit(result)
# Plot results
# plt.figure()
# plt.plot(impedance.real, -impedance.imag, 'o', label = 'data')
# plt.plot(fit.real, -fit.imag, label = 'model')
# plt.legend()
# plt.axis('equal')
# plt.show()

plt.figure()
plt.plot(impedance.real, -impedance.imag, 'o', label = 'data')
plt.plot(fit.real, -fit.imag, label = 'model')
plt.legend()
plt.axis('equal')
plt.show()