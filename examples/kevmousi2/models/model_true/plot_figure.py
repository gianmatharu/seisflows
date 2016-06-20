from seisflows.seistools.ewf2d import Par
import numpy as np
import matplotlib.pyplot as plt
from seisflows.tools.graphics import get_tick_vectors
from mpl_toolkits.axes_grid1 import make_axes_locatable

input_file = "../../solver/INPUT/par_template.cfg"

# Read parameter file
p = Par()
p.read_par_file(input_file)

# Process information
npad = p.npad
nx = p.nx
nz = p.nz
dx = p.dx
print(nx, nz)

# Get collected arrays
vp = np.fromfile("vp.bin", dtype='float32')
vs = np.fromfile("vs.bin", dtype='float32')
vp = vp.reshape((nz,nx))
vs = vs.reshape((nz,nx))

plt.set_cmap('seismic_r')

fig, ax = plt.subplots()
im = ax.imshow(vp)

ix, x = get_tick_vectors(nx, dx, 1000.0)
iz, z = get_tick_vectors(nz, dx, 500.0)
im.xticks(ix, x[ix] / 1000.0)
im.yticks(iz, z[iz] / 1000.0)
plt.xlabel('Distance (km)')
plt.ylabel('Depth (km)')

divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
cbar = plt.colorbar(im, cax=cax)
cbar.set_ticks([vp.min(), 0.5 * (vp.max() + vp.min()), vp.max()])
cbar.set_ticklabels([vp.min(), 0.5 * (vp.max() + vp.min()), vp.max()])


plt.tight_layout()
plt.show()
