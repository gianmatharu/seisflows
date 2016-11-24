import matplotlib.pyplot as plt
from seisflows.seistools.model import build_perturbation_model, build_homogeneous_model

nx = 100
nz = 50
rho_homo = 1000
vp_homo = 3000
vs_homo = vp_homo / 1.76
perc = 50
sigma = 3

vp = build_perturbation_model(nx, nz, vp_homo, [(50, 25, sigma, perc)])
#vs = build_homogeneous_model(nx, nz, vs_homo)
#rho = build_homogeneous_model(nx, nz, rho_homo)
vs = build_perturbation_model(nx, nz, vs_homo, [(25, 50, sigma, 0)])
rho = build_perturbation_model(nx, nz, rho_homo, [(25, 50, sigma, 0)])

plt.subplot(3, 1, 1)
plt.imshow(vp)
plt.subplot(3, 1, 2)
plt.imshow(vs)
plt.subplot(3, 1, 3)
plt.imshow(rho)
plt.show()

vp.astype('float32').tofile('model_true/vp.bin')
vs.astype('float32').tofile('model_true/vs.bin')
rho.astype('float32').tofile('model_true/rho.bin')

#vp_init = build_perturbation_model(nx, nz, vp_homo, [(150, 150, sigma, 0.6*perc)])
vp_init = build_homogeneous_model(nx, nz, vp_homo)
vs_init = build_homogeneous_model(nx, nz, vs_homo)
rho_init = build_homogeneous_model(nx, nz, rho_homo)

vp_init.astype('float32').tofile('model_init/vp.bin')
vs_init.astype('float32').tofile('model_init/vs.bin')
rho_init.astype('float32').tofile('model_init/rho.bin')
