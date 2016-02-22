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

vs = vp / ratio
# Ensure CPML layers have homogenous velocity
if (p.use_cpml_left):
    vp[:, 0: p.ncpml] = np.floor(vp.mean())

if (p.use_cpml_right):
    vp[:, nx - p.ncpml:nx] = np.floor(vp.mean())

if (p.use_cpml_top):
    vp[0:p.ncpml, :] = np.floor(vp.mean())

if (p.use_cpml_bottom):
    vp[nz-p.ncpml:nz, :] = np.floor(vp.mean())


#vp[60:160, 210:310] = 3600

r = 1500
p1 = [2175, 5925]
p2 = [5925, 2175]

x = np.linspace(0, nx * p.dx, nx)
z = np.linspace(0, nz * p.dz, nz)


for i in range(0, nz):
    for j in range(0, nx):
        if dist(x[j], z[i], p1[0], p1[1]) < r:
            vp[i][j] = 3300 * 1.1

for i in range(0, nz):
    for j in range(0, nx):
        if dist(x[j], z[i], p2[0], p2[1]) < r:
            vp[i][j] = 3300 * 0.9


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
