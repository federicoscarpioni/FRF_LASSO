from data_loaders import load_sequential
from slider_fit_plot import plot_impedance_set
import matplotlib.pyplot as plt
import numpy as np

from visualize_fit_and_error import plot_impedance_set_with_errors_weights

path = "data/results_for_thesis/2601291245_Cell015_fit_ratpoly9_wfmodulox0.5_pene-12/cycle_3_sequence_4"

model, weights, results, impedance_set, fits, frequencies = load_sequential(path)

# Plot results
plt.figure()
for p in results[0].params:
    param_values = [results[i].params[p] for i in range(len(results))]
    # errors = param_values = [results[i].params[p].stderr for i in range(len(results))]
    # indexes = np.arange(1, len(param_values)+1,1)
    plt.plot(param_values)
    # plt.errorbar(indexes, param_values, errors)
# plot_impedance_set(impedance_set, fits)

plot_impedance_set_with_errors_weights(impedance_set, 1/weights, fits, results, frequencies)

plt.show()

