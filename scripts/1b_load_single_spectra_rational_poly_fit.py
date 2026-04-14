import matplotlib.pyplot as plt
import numpy as np
import lmfit
from data_loaders import load

path = './data/results/2510061349_single_spectra_test'

model, result, impedance, fit = load(path)

lmfit.report_fit(result)
# Plot results
plt.figure()
plt.plot(impedance.real, -impedance.imag, 'o', label = 'data')
plt.plot(fit.real, -fit.imag, label = 'model')
plt.legend()
plt.axis('equal')
plt.show()