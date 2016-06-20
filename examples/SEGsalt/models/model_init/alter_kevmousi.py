import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from seisflows.tools.array import gridsmooth, readgrid
from seisflows.seistools.ewf2d import Par
from scipy.ndimage.filters import gaussian_filter

input_file = "../../solver/INPUT/par_template.cfg"
p = Par()
p.read_par_file(input_file)

# original model parameters from Kevin's default model.
dx = 20
ratio = 1.76
vp = readgrid('vp.bin', p.nx, p.nz, dtype='float32')
#vp = gaussian_filter(vp, sigma=2)
vp = gridsmooth(vp, 10)
# Add padded layer
rho_homo = 2600

vs = vp / ratio
rho_out = rho_homo * np.ones((p.nx, p.nz))

plt.set_cmap('jet_r')
plt.subplot(2, 1, 1)
plt.imshow(vp)
plt.subplot(2, 1, 2)
plt.imshow(vs)
plt.show()

vp.astype("float32").tofile("vps.bin")
vs.astype("float32").tofile("vss.bin")
rho_out.astype("float32").tofile("rho.bin")

