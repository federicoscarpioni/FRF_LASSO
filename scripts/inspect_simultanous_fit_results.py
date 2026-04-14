import numpy as np
import matplotlib.pyplot as plt
from data_loaders import load_simultaneous
from simultaneous_routines import extract_global_param_evolution
from slider_fit_plot import plot_impedance_set

basic_path = "./data/results/2510130109_simultaneus_cell10_jaxs/charge/charge_Model(rational_poly7)_reg0.01_smt0.01"

cycle = 1

fit_folder = basic_path + f"/cycle_{cycle}"#_s0_reg1e-4_smt1e-3"
model, result, impedances, fits = load_simultaneous(fit_folder)
params_list = extract_global_param_evolution(result, model.param_names,len(impedances))
plot_impedance_set(impedances,fits)