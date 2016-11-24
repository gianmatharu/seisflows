import matplotlib.pyplot as plt
from seisflows.seistools.model import build_perturbation_model, build_homogeneous_model
from seisflows.tools.array import readgrid
from seisflows.tools.dsp import apply_isotropic_smoothing
from numpy.random import randn
import numpy as np

nx = 100
nz = 50
rho_homo = 1000
vp_homo = 3000
vs_homo = vp_homo / 1.76
perc = 1
sigma = 2

vp = readgrid('model_init/vp.bin', nx, nz, dtype='float32')
vs = readgrid('model_init/vs.bin', nx, nz, dtype='float32')
rho = readgrid('model_init/rho.bin', nx, nz, dtype='float32')

#vp_pert = build_perturbation_model(nx, nz, vp_homo, [(75, 65, sigma, -0.8*perc), (75, 25, sigma, perc)], model=vp)
#rho_pert = build_perturbation_model(nx, nz, rho_homo, [(75, 65, sigma, -0.8*perc), (200, 25, sigma, perc)], model=rho)
#vs_pert = build_perturbation_model(nx, nz, vs_homo, [(75, 25, sigma, -0.8*perc), (25, 25, sigma, perc)], model=vs)

nxr, nzr = (290, 290)
sigmar = 0.1 * vp.mean() / 2.355

vp_pert = vp
vs_pert = vs
rho_pert = rho + 1

plt.subplot(3, 1, 1)
plt.imshow(vp_pert)
plt.subplot(3, 1, 2)
plt.imshow(vs_pert)
plt.subplot(3, 1, 3)
plt.imshow(rho_pert)
plt.show()

vp_pert.astype('float32').tofile('model_pert/vp.bin')
vs_pert.astype('float32').tofile('model_pert/vs.bin')
rho_pert.astype('float32').tofile('model_pert/rho.bin')
