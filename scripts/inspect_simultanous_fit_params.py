import numpy as np
import matplotlib.pyplot as plt
from data_loaders import load_simultaneous
from simultaneous_routines import extract_global_param_evolution

basic_path = "data/results/2510092152_simultaneus_cell10_not_norm_previous/charge_Model(rational_poly5)_reg0.001_smt0.01"

plt.figure()
cycles = range(1,20,1)

cm = plt.get_cmap('jet')
cmmap = [cm(1.*i/(cycles[-1]-cycles[0])) for i in cycles]


for i,c in enumerate(cycles):
    fit_folder = basic_path + f"/cycle_{c}"
    model, result, impedances, fits = load_simultaneous(fit_folder)
    params_list = extract_global_param_evolution(result, model.param_names,len(impedances))
    plt.plot(params_list['a1'], color =cmmap[i], label = c)
plt.legend()
plt.show()