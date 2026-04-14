import json, lmfit
import numpy as np
import matplotlib.pyplot as plt
from routines import multiple_start_rational_poly_fit
from rational_poly_models import rational_poly6
from obj_functions import complex_least_square_regl1
import fit_statistics
from data_savers import save_multistart
from slider_fit_plot import plot_impedance_set

# Import one spectra to fit
folder = 'C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_010/2507281144PanCoin2025_010_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz/pico_aquisition/cycle_10_sequence_0'
impedance = np.load(folder+'/impedance_total.npy')[1:-3,200]
metadata_file = open("C:/Users/fscarp/ownCloud/Data/2025/Coin_MnO2_2025/PanCoin2025_009/2507281125PanCoin2025_009_deisCCCV_x30_100s_window_fmin10Hz_fs_50Hz/metadata_deis_exp.json")
frequencies = np.array( json.load(metadata_file)['Frequencies multisine (Hz)'] )[1:-3]

# Fit
model = lmfit.Model(rational_poly6)
obj_args = {'reg_factor':1e-12}
results, fits = multiple_start_rational_poly_fit(
    frequencies,
    impedance,
    model,
    complex_least_square_regl1,
    obj_args,
    n_trials=10,
    seed=42,
    min = 1e-4,
    max = 1e4
)
stats = fit_statistics.chi_square_statistics(results)
fit_statistics.print_chi_square_analysis(stats)
# fit_statistics.visualize_chi_square_distribution(stats)
impedance_list= [impedance]*len(fits)
plot_impedance_set(impedance_list,fits)

saving_path = "./data/results/2510061924_multistart_single_spectra_test"
save_multistart(
    saving_path,
    model,
    results,
    impedance,
    fits, 
)