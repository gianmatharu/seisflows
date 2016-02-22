import sys

import time
import numpy as np
import matplotlib.pyplot as plt
from fwpy.seistools.plotutils import Par

input_file = "../../solver/INPUT/par.cfg"
rank_file = "ranks.txt"

# Read parameter file
p = Par()
p.read_par_file(input_file)
clim = 1

# Process information
nproc = p.nprocx * p.nprocz
#coords = np.zeros((nproc, 3))
coords = np.loadtxt(rank_file, dtype=int)
npad = p.npad
nx = p.nx
nz = p.nz
nx_l = (p.nx / p.nprocx) + 2 * npad
nz_l = (p.nz / p.nprocz) + 2 * npad

plt.ion()
plt.show()

for i in range(0, p.ntimesteps, p.output_interval):
    # Get collected arrays
    vx_file = "snapshot_vx_{:05d}.bin".format(i)
    vz_file = "snapshot_vz_{:05d}.bin".format(i)
    ux = np.fromfile(vx_file, dtype='float32')
    uz = np.fromfile(vz_file, dtype='float32')
    ux = ux.reshape((nz, nx));
    uz = uz.reshape((nz, nx));

    plt.subplot(2, 1, 1)
    plt.imshow(ux)
    plt.clim(-clim, clim)
    plt.title("Vx - Time = {:3f}".format(p.dt*i))
    plt.subplot(2, 1, 2)
    plt.imshow(uz)
    plt.clim(-clim, clim)
    plt.title("Vz - Time = {:3f}".format(p.dt*i))
    plt.draw()
    time.sleep(0.5)
    plt.pause(0.0001)
