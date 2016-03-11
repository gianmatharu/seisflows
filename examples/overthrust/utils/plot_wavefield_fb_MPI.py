from fwpy.seistools.plotutils import Par

import time
import numpy as np
import matplotlib.pyplot as plt

input_file = "../../solver/INPUT/par.cfg"
rank_file = "ranks.txt"


# Read parameter file
p = Par()
p.read_par_file(input_file)
clim = 1000
rlim = 1e-4
# Process information
nproc = p.nprocx * p.nprocz
coords = np.loadtxt(rank_file, dtype=int)

npad = p.npad
nx = p.nx
nz = p.nz

plt.ion()
plt.show()


for i in range(p.ntimesteps, 0, -p.output_interval):
    # Get collected arrays
    vx_file = "snapshot_vx_{:05d}.bin".format(i)
    vz_file = "snapshot_vz_{:05d}.bin".format(i)
    ux = np.fromfile(vx_file, dtype='float32')
    uz = np.fromfile(vz_file, dtype='float32')
    ux = ux.reshape((nz, nx))
    uz = uz.reshape((nz, nx))

    b_vx_file = "snapshot_b_vx_{:05d}.bin".format(p.ntimesteps-i)
    b_vz_file = "snapshot_b_vz_{:05d}.bin".format(p.ntimesteps-i)
    b_ux = np.fromfile(b_vx_file, dtype='float32')
    b_uz = np.fromfile(b_vz_file, dtype='float32')
    b_ux = b_ux.reshape((nz, nx))
    b_uz = b_uz.reshape((nz, nx))

    plt.subplot(3, 2, 1)
    plt.imshow(ux)
    plt.clim(-clim, clim)
    plt.title("Vx - Time = {:3f}".format(p.dt*i))
    plt.subplot(3, 2, 2)
    plt.imshow(uz)
    plt.clim(-clim, clim)
    plt.title("Vz - Time = {:3f}".format(p.dt*i))

    plt.subplot(3, 2, 3)
    plt.imshow(b_ux)
    plt.clim(-clim, clim)
    plt.title("Reconstructed Vx - Time = {:3f}".format(p.dt*i))
    plt.subplot(3, 2, 4)
    plt.imshow(b_uz)
    plt.clim(-clim, clim)
    plt.title("Reconstructed Vz - Time = {:3f}".format(p.dt*i))

    plt.subplot(3, 2, 5)
    plt.imshow(ux-b_ux)
    plt.clim(-rlim, rlim)
    plt.title("Residual Vx")

    plt.subplot(3, 2, 6)
    plt.imshow(uz-b_uz)
    plt.clim(-rlim, rlim)
    plt.title("Residual Vz")

    plt.draw()
    time.sleep(0.5)
    plt.pause(0.0001)
