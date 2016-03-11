from fwpy.seistools.plotutils import Par

import time
import numpy as np
import matplotlib.pyplot as plt

input_file = "../../solver/INPUT/par.cfg"

# Read parameter file
p = Par()
p.read_par_file(input_file)

# Process information
nproc = p.nprocx * p.nprocz

npad = p.npad
nx = p.nx
nz = p.nz

k_alpha = np.fromfile("alpha_kernel.bin", dtype='float32')
k_beta = np.fromfile("beta_kernel.bin", dtype='float32')
k_rhop = np.fromfile("rhop_kernel.bin", dtype='float32')
k_alpha = k_alpha.reshape((nz, nx))
k_beta = k_beta.reshape((nz, nx))
k_rhop = k_rhop.reshape((nz, nx))

k_lambda = np.fromfile("lambda_kernel.bin", dtype='float32')
k_mu = np.fromfile("mu_kernel.bin", dtype='float32')
k_rho = np.fromfile("rho_kernel.bin", dtype='float32')
k_lambda = k_lambda.reshape((nz, nx))
k_mu = k_mu.reshape((nz, nx))
k_rho = k_rho.reshape((nz, nx))

clim = 1e-3
plt.subplot(3, 2, 1)
plt.imshow(k_alpha)
plt.title('Vp kernel')
plt.clim(-clim, clim)
plt.subplot(3, 2, 2)
plt.imshow(k_beta)
plt.title('Vs kernel')
plt.clim(-clim, clim)
plt.subplot(3, 2, 3)
plt.imshow(k_rhop)
plt.title('rho_p kernel')
plt.clim(-clim, clim)

plt.subplot(3, 2, 4)
plt.imshow(k_lambda)
plt.clim(-1e-11, 1e-11)
plt.subplot(3, 2, 5)
plt.imshow(k_mu)
plt.clim(-1e-11, 1e-11)
plt.subplot(3, 2, 6)
plt.imshow(k_rho)
plt.clim(-clim, clim)




plt.show()
