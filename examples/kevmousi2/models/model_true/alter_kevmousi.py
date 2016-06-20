import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from seisflows.tools.array import gridsmooth
from seisflows.seistools.ewf2d import Par

input_file = "../../solver/INPUT/par_template.cfg"
p = Par()
p.read_par_file(input_file)

# original model parameters from Kevin's default model.
dx = 10
dz = 5
x = np.arange(0, 7000, dx)
z = np.arange(0, 3000, dz)
ratio = 1.76
vp = np.loadtxt("vel.txt", dtype=float)

# Apply velocity scaling on model.

#vp *= 2
xx, zz = np.meshgrid(x, z)
f_vp = interpolate.interp2d(x, z, vp, kind='cubic')

# Interpolate onto new grid.
dx = 20
xn = np.arange(-2000, 8000, dx)
zn = np.arange(500, 3100, dx)
vpn = f_vp(xn, zn)
nz = vpn.shape[0]
nx = vpn.shape[1]

for i in range(nz):
    for j in range(nx):
        if vpn[i, j] < 1580.0:
            vpn[i, j] = 1580.0

#vpn = gridsmooth(vpn, 20)
# Add padded layer
vp_out = vpn

rho_homo = 2600

vs_out = vp_out / ratio
#vs_out = gridsmooth(vs_out, 15)
rho_out = rho_homo * np.ones((p.nx, p.nz))

plt.subplot(3, 1, 1)
plt.imshow(vp)
plt.subplot(3, 1, 2)
plt.imshow(vpn)
plt.subplot(3, 1, 3)
plt.imshow(vp_out)
plt.show()

vp_out.astype("float32").tofile("vp.bin")
vs_out.astype("float32").tofile("vs.bin")
rho_out.astype("float32").tofile("rho.bin")

