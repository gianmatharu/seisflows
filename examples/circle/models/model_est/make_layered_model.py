from fwpy.seistools.plotutils import Par
import matplotlib.pyplot as plt
import numpy as np

def dist(x1, z1, x2, z2):
    return np.sqrt((x1 - x2) * (x1 - x2) + (z1 - z2) * (z1 - z2))


input_file = "../../solver/INPUT/par_template.cfg"

p = Par()
p.read_par_file(input_file)

rho_homo = 2600
ratio = 1.76
nx = p.nx
nz = p.nz

vp = 3300 * np.ones((nz, nx))

# Ensure CPML layers have homogenous velocity
if (p.use_cpml_left):
    vp[:, 0: p.ncpml] = np.floor(vp.mean())

if (p.use_cpml_right):
    vp[:, nx - p.ncpml:nx] = np.floor(vp.mean())

if (p.use_cpml_top):
    vp[0:p.ncpml, :] = np.floor(vp.mean())

if (p.use_cpml_bottom):
    vp[nz-p.ncpml:nz, :] = np.floor(vp.mean())



#p1 = [2000, 2000]
#p2 = [6000, 2000]

#x = np.linspace(0, nx * p.dx, nx)
#z = np.linspace(0, nz * p.dz, nz)

#vp[110:310, 310:510] = 3700

#for i in range(0, nz):
#    for j in range(0, nx):
#        if dist(x[j], z[i], p1[0], p1[1]) < 1000:
#            vp[i][j] = 3700

#for i in range(0, nz):
#    for j in range(0, nx):
#        if dist(x[j], z[i], p2[0], p2[1]) < 1000:
#            vp[i][j] = 3000


vs = vp / ratio
rho = rho_homo * np.ones((nz, nx))

plt.subplot(3, 1, 1)
plt.imshow(vp)
plt.subplot(3, 1, 2)
plt.imshow(vs)
plt.subplot(3, 1, 3)
plt.imshow(rho)
plt.show()

vp.astype("float32").tofile("vp.bin")
vs.astype("float32").tofile("vs.bin")
rho.astype("float32").tofile("rho.bin")
