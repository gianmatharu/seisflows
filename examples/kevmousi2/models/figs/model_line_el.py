import sys
import numpy as np
from seisflows.tools.array import readgrid
import matplotlib.pyplot as plt

nx = 500
nz = 130
dx = 20

z = np.arange(0, nz * dx, dx)
xline = float(sys.argv[1])
xline1 = float(sys.argv[2])
xline2 = float(sys.argv[3])

iline = int(xline / dx)
iline1 = int(xline1 / dx)
iline2 = int(xline2 / dx)

vp_true = readgrid('true_vp.bin', nx, nz, dtype='float32')
vp_start = readgrid('start_vp.bin', nx, nz, dtype='float32')
vp_final = readgrid('cg_el_vp.bin', nx, nz, dtype='float32')

fig = plt.figure(figsize=(12,5))
plt.subplot(1, 3, 1)
line1, = plt.plot(vp_true[:, iline], z, label='True')
line2, = plt.plot(vp_start[:, iline], z, label='Initial')
line3, = plt.plot(vp_final[:, iline], z, label='Final')
plt.legend(handles=[line1, line2, line3])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.xlim(1000, 3200)
plt.ylim(0, 2600)
plt.title('x = {}'.format(xline))
plt.gca().invert_yaxis()

plt.subplot(1, 3, 2)
line1, = plt.plot(vp_true[:, iline1], z, label='True')
line2, = plt.plot(vp_start[:, iline1], z, label='Initial')
line3, = plt.plot(vp_final[:, iline1], z, label='Final')
plt.legend(handles=[line1, line2, line3])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.xlim(1000, 3200)
plt.ylim(0, 2600)
plt.title('x = {}'.format(xline1))
plt.gca().invert_yaxis()

plt.subplot(1, 3, 3)
line1, = plt.plot(vp_true[:, iline2], z, label='True')
line2, = plt.plot(vp_start[:, iline2], z, label='Initial')
line3, = plt.plot(vp_final[:, iline2], z, label='Final')
plt.legend(handles=[line1, line2, line3])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Depth (m)')
plt.xlim(1000, 3200)
plt.ylim(0, 2600)
plt.title('x = {}'.format(xline2))
plt.gca().invert_yaxis()

fig.tight_layout()

plt.savefig('section_el_vp.eps', format='eps', bbox_inches='tight')
plt.show()
