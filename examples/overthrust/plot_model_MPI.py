from fwpy.seistools.plotutils import Par
import time
import numpy as np
import matplotlib.pyplot as plt

input_file = "../../solver/INPUT/par_template.cfg"

# Read parameter file
p = Par()
p.read_par_file(input_file)

# Process information
nproc = p.nprocx * p.nprocz

npad = p.npad
nx = p.nx
nz = p.nz
print(nx, nz)

# Get collected arrays
vp = np.fromfile("vp.bin", dtype='float32')
vs = np.fromfile("vs.bin", dtype='float32')
vp = vp.reshape((nz,nx))
vs = vs.reshape((nz,nx))

plt.subplot(2, 1, 1)
plt.imshow(vp)
plt.title("Vp")
plt.subplot(2, 1, 2)
plt.imshow(vs)
plt.title("Vs")

plt.show()
