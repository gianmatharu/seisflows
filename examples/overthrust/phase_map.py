import numpy as np
import matplotlib.pyplot as plt
from seisflows.seistools.ewf2d import Par
from os.path import join
from obspy import read

from obspy import read
from seisflows.seistools.susignal import sdamping
from seisflows.seistools.ewf2d import event_dirname
from seisflows.tools.graphics import plot_obs_section
from os.path import join
from scipy.signal import gaussian
from cmath import phase
import scipy.signal as signal
from numpy.fft import fft, ifft



basedir = '/home/gian/Desktop/seisflows/examples/kevmousi2'
parfile = join(basedir, 'solver/INPUT/par_template.cfg')

p = Par()
p.read_par_file(parfile)

basedir = '/home/gian/Desktop/seisflows/examples/kevmousi2/phasemap_data_240src/solver'

n = 240
nr = 249
dt = p.dt
nt = p.ntimesteps
Fs = 1/dt
k = np.arange(nt)
freq = (Fs / nt) * k
f0 = 10.0
ifq = 11
print freq[ifq]
t = np.arange(0, nt*dt, dt)
twin = 1.0

pmap = np.zeros((n, nr))

for itask in range(n):

    print ('Running {} of {}'.format(itask, n))
    dfile = join(basedir, event_dirname(itask + 1), 'traces/obs', 'Ux_data.su')
    sfile = join(basedir, event_dirname(itask + 1), 'traces/syn', 'Ux_data.su')

    # Read data and synthetic
    d = read(dfile, format='SU')
    s = read(sfile, format='SU')

    # apply windowing
    dd = sdamping(d, twin)
    sd = sdamping(s, twin)

    for ir in range(nr):
        Fd = fft(dd[ir].data)
        Fs = fft(sd[ir].data)
        pmap[itask, ir] = phase(Fs[ifq] * np.conj(Fd[ifq]))


plt.set_cmap('seismic_r')


pm = np.zeros((500, 500))
xs = np.arange(260, 9840, 40)
dx = 20.0
xrec = np.arange(40, 10000, 40)
for itask in range(n):
    ixs = int(xs[itask] / dx)
    for irec in range(nr):
        ixr = int(xrec[irec]/ dx)
        pm[ixs, ixr] = pmap[itask, irec]

plt.subplot(1, 2, 1)
plt.imshow(pmap)

plt.subplot(1, 2,2)
plt.imshow(pm)
plt.show()
