import numpy as np
from os.path import join
from seisflows.tools.array import readgrid
from seisflows.seistools.psf import generate_uniform_spikes
import matplotlib.pyplot as plt

# set parameters
nx, nz = (300, 100)
par = 'rho'
pos = (10, 10)
dx, dz = (20, 20)
mt_path = 'models/model_true'
mi_path = 'models/model_init'
mp_path = 'models/model_pert'

# load models and display
mt = readgrid(join(mt_path, par+'.bin'), nx, nz, dtype='float32')
mi = readgrid(join(mi_path, par+'.bin'), nx, nz, dtype='float32')

plt.set_cmap('seismic_r')
plt.subplot(1, 2, 1)
plt.imshow(mt)
plt.subplot(1, 2, 2)
plt.imshow(mi)

#mp = generate_uniform_spikes(mi, pos, dx, dz)
mp = mi + 1
mp.astype('float32').tofile(join(mp_path, par+'.bin'))
plt.imshow(mp)
plt.show()
