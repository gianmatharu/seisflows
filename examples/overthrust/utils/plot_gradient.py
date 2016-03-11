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

k_alpha = np.fromfile("vp_kernel.bin", dtype='float32')
k_beta = np.fromfile("vs_kernel.bin", dtype='float32')
k_alpha = k_alpha.reshape((nz, nx))
k_beta = k_beta.reshape((nz, nx))

clim = 1e-3
plt.subplot(2, 1, 1)
plt.imshow(k_alpha)
plt.title('Vp kernel')
plt.clim(-clim, clim)
plt.subplot(2, 1, 2)
plt.imshow(k_beta)
plt.title('Vs kernel')
plt.clim(-clim, clim)


plt.show()
