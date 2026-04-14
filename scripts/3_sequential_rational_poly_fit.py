import json, lmfit
import numpy as np
import matplotlib.pyplot as plt
from slider_fit_plot import plot_impedance_set
from rational_poly_models import rational_poly5
from obj_functions import complex_least_square_regl1
from routines import sequential_rational_poly_fit, rational_poly_fit
from data_savers import save_sequential

# Import one spectra to fit
cell_folder = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_015/2509151200PanCoin2025_015_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz'
# cell_folder = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_013/2508191421PanCoin2025_013_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz'
# cell_folder = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_009/2507281125PanCoin2025_009_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz'
experiment_path = cell_folder + '/pico_aquisition/cycle_5_sequence_4'
impedance = np.load(experiment_path+'/impedance_total.npy')[1:-3,3:-2:1]
impedance_set = [impedance[:,i] for i in range(impedance.shape[1])]
metadata_file = open(cell_folder+"/metadata_deis_exp.json")
frequencies = np.array( json.load(metadata_file)['Frequencies multisine (Hz)'] )[1:-3]

# Chose folder to save results (path created autom.)
saving_path = "./data/results_for_thesis/2601261721_Cell015_cycle_1_sequence_5_ratpoly5_wfmodulox0.5_smthe-12"

# Create model and initial parameters
model = lmfit.Model(rational_poly5)
params = lmfit.Parameters()
np.random.seed(42)
for key in model.param_names:
    params[key] = lmfit.Parameter(name=key, value=np.random.lognormal(-6, 6), min=1e-6, max = 1e6)

# Set weighting factors per frequency point
# weights = (frequencies)**(-1)
# weights = (frequencies)**(-0.1)
# weights = np.abs((impedance_set[2]))**(-2)
weights = np.abs((impedance_set[2]))**(-0.5)
# weights = 1/np.log(impedance_set[2])

# Fit
obj_args={
    'reg_factor': 1e-6,
}
# Fit the spectra in the middle
print('Fit the spectra in the middle')
middle_index = len(impedance_set)//2
# Fit impedance
result, fit = rational_poly_fit(
    frequencies,
    impedance_set[middle_index],
    model,
    np.abs(impedance_set[middle_index])**(-0.5),
    params,#global_result.params,
    complex_least_square_regl1,
    obj_args,
)
# Fit admittance
# result, fit = rational_poly_fit(
#     frequencies,
#     1/impedance_set[middel_index],
#     model,
#     params,#global_result.params,
#     complex_least_square_regl1,
#     obj_args,
# )
# fit = 1/fit
print("Starting simultanous fit.")
# Fit impedance
results,fits = sequential_rational_poly_fit(
    frequencies,
    impedance_set,
    model,
    [np.abs(impedance)**(-0.5) for impedance in impedance_set],
    result.params,
    complex_least_square_regl1,
    obj_args,
)
# Fit admittance
# admittance_set = [1/impedance for impedance in impedance_set]
# results,fits = sequential_rational_poly_fit(
#     frequencies,
#     admittance_set,
#     model,
#     result.params,
#     complex_least_square_regl1,
#     obj_args,
# )
# fits = [1/fit for fit in fits]


save_sequential(
    saving_path,
    model,
    weights,
    results,
    impedance_set,
    fits,
    frequencies,
)

# Plot results
plt.figure()
for p in params:
    param_values = [results[i].params[p] for i in range(len(results))]
    plt.plot(param_values)
plot_impedance_set(impedance_set, fits)
plt.show()