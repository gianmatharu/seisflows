from fwpy.seistools.plotutils import Par

import time
import numpy as np
import matplotlib.pyplot as plt

input_file = "../../solver/INPUT/par.cfg"
rank_file = "ranks.txt"

# Read parameter file
p = Par()
p.read_par_file(input_file)

# Process information
nproc = p.nprocx * p.nprocz
coords = np.loadtxt(rank_file, dtype=int)

npad = p.npad
nx = p.nx
nz = p.nz

k_alpha = np.fromfile("alpha_kernel.bin", dtype='float32')
k_beta = np.fromfile("beta_kernel.bin", dtype='float32')
k_alpha = k_alpha.reshape((nz, nx))
k_beta = k_beta.reshape((nz, nx))

plt.subplot(2, 1, 1)
plt.imshow(k_alpha)
plt.clim(-1e-5, 1e-5)
plt.subplot(2, 1, 2)
plt.imshow(k_beta)
plt.clim(-1e-5, 1e-5)
plt.show()
