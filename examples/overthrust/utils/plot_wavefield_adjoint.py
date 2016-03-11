from fwpy.seistools.plotutils import Par

import time
import numpy as np
import matplotlib.pyplot as plt

input_file = "../../solver/INPUT/par.cfg"
rank_file = "ranks.txt"


# Read parameter file
p = Par()
p.read_par_file(input_file)
clim = 100
rlim = 1e-6
# Process information
nproc = p.nprocx * p.nprocz
coords = np.loadtxt(rank_file, dtype=int)

npad = p.npad
nx = p.nx
nz = p.nz

plt.ion()
plt.show()


for i in range(0, p.ntimesteps, p.output_interval):
    # Get collected arrays
    vx_file = "snapshot_vx_{:05d}.bin".format(p.ntimesteps-i)
    vz_file = "snapshot_vz_{:05d}.bin".format(p.ntimesteps-i)
    ux = np.fromfile(vx_file, dtype='float32')
    uz = np.fromfile(vz_file, dtype='float32')
    ux = ux.reshape((nz, nx))
    uz = uz.reshape((nz, nx))

    b_vx_file = "snapshot_b_vx_{:05d}.bin".format(i)
    b_vz_file = "snapshot_b_vz_{:05d}.bin".format(i)
    b_ux = np.fromfile(b_vx_file, dtype='float32')
    b_uz = np.fromfile(b_vz_file, dtype='float32')
    b_ux = b_ux.reshape((nz, nx))
    b_uz = b_uz.reshape((nz, nx))


    alpha_file = "alpha_{:05d}.bin".format(i)
    beta_file = "beta_{:05d}.bin".format(i)
    alpha = np.fromfile(alpha_file, dtype='float32')
    beta = np.fromfile(beta_file, dtype='float32')
    alpha = b_ux.reshape((nz, nx))
    beta = b_uz.reshape((nz, nx))

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
    plt.clim(-1e-7, 1e-7)
    plt.title("Reconstructed Vx - Time = {:3f}".format(p.dt*i))
    plt.subplot(3, 2, 4)
    plt.imshow(b_uz)
    plt.clim(-1e-7, 1e-7)
    plt.title("Reconstructed Vz - Time = {:3f}".format(p.dt*i))

    plt.subplot(3, 2, 5)
    plt.imshow(alpha)
    plt.clim(-rlim, rlim)
    plt.title('alpha_kernel')
    plt.subplot(3, 2, 6)
    plt.imshow(beta)
    plt.clim(-rlim, rlim)
    plt.title('beta kernel')

    plt.draw()
    time.sleep(0.5)
    plt.pause(0.0001)
