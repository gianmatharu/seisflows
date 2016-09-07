import numpy as np
from seisflows.seistools.susignal import compute_amplitude_spectrum

x = np.loadtxt('stf.txt')
t = x[:, 0]
y = x[:, 1]
dt = t[1]-t[0]
Fs = 1/dt

compute_amplitude_spectrum(t, y, Fs, verbose=True)