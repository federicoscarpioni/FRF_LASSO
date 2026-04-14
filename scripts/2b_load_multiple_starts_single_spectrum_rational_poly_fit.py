import fit_statistics
from data_loaders import load_multistart
from slider_fit_plot import plot_impedance_set

# path = "./data/results/2510061924_multistart_single_spectra_test"
path = "experiments/2510070920multiple_starting_points_cell10_ratpoly6/data/cycle5_sequence0/spectrum  8/reg_factor1e-03"
model, results, impedance, fits = load_multistart(path)

stats = fit_statistics.chi_square_statistics(results)
fit_statistics.print_chi_square_analysis(stats)
impedance_list= [impedance[:,8]]*len(fits)
plot_impedance_set(impedance_list,fits)