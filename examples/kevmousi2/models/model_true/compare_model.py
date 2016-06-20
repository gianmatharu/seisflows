import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from seisflows.tools.array import gridsmooth, readgrid
from seisflows.seistools.ewf2d import Par

input_file = "../../solver/INPUT/par_template.cfg"
p = Par()
p.read_par_file(input_file)

# original model parameters from Kevin's default model.
vp = readgrid('vp.bin', p.nx, p.nz, dtype='float32')
vs = readgrid('vs.bin', p.nx, p.nz, dtype='float32')

vpi = readgrid('../model_init/vp.bin', p.nx, p.nz, dtype='float32')
vsi = readgrid('../model_init/vs.bin', p.nx, p.nz, dtype='float32')

#plt.set_cmap('seismic_r')
plt.subplot(2, 3, 1)
plt.imshow(vp, aspect='auto')
plt.subplot(2, 3, 2)
plt.imshow(vpi, aspect='auto')
plt.subplot(2, 3, 3)
#plt.imshow(gridsmooth(vpi - vp, 20), aspect='auto')
plt.imshow(vpi - vp, aspect='auto')
plt.colorbar()
plt.subplot(2, 3, 4)
plt.imshow(vs, aspect='auto')
plt.subplot(2, 3, 5)
plt.imshow(vsi, aspect='auto')
plt.subplot(2, 3, 6)
#plt.imshow(gridsmooth(vsi - vs, 10), aspect='auto')
plt.imshow(vsi - vs, aspect='auto')
plt.colorbar()
plt.show()

