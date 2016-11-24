import sys
import matplotlib.pyplot as plt
from seisflows.seistools.model import build_perturbation_model, build_homogeneous_model
from seisflows.tools.array import readgrid
from seisflows.tools.dsp import apply_isotropic_smoothing
from numpy.random import randn
import numpy as np

def add_delta_perturbation(model, x, y):
    for i in range(len(y)):
        for j in range(len(x)):
            model[y[i], x[j]] += 10

    return model
par = sys.argv[1]
dx = int(sys.argv[2])
dz = int(sys.argv[3])

nx = 100
nz = 50

x = np.arange(dx, nx, dx)
z = np.arange(dz, nz, dz)

vp = readgrid('model_init/vp.bin', nx, nz, dtype='float32')
vs = readgrid('model_init/vs.bin', nx, nz, dtype='float32')
rho = readgrid('model_init/rho.bin', nx, nz, dtype='float32')

vp_pert = np.copy(vp)
vs_pert = np.copy(vs)
rho_pert = np.copy(rho)

if par == 'vp':
    vp_pert = add_delta_perturbation(vp_pert, x, z)
if par == 'vs':
    vs_pert = add_delta_perturbation(vs_pert, x, z)
if par == 'rho':
    print('here')
    rho_pert = add_delta_perturbation(rho_pert, x, z)

plt.subplot(3, 1, 1)
plt.imshow(vp_pert)
plt.subplot(3, 1, 2)
plt.imshow(vs_pert)
plt.subplot(3, 1, 3)
plt.imshow(rho_pert)

plt.figure(2)
plt.imshow(rho_pert - rho)
plt.clim(0, 1)
plt.set_cmap('Greys')
plt.show()

vp_pert.astype('float32').tofile('model_pert/vp.bin')
vs_pert.astype('float32').tofile('model_pert/vs.bin')
rho_pert.astype('float32').tofile('model_pert/rho.bin')
