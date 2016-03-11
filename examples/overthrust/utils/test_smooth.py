import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import misc
from fwpy.tools.array import gridsmooth

from fwpy.seistools.plotutils import Par

p = Par()
p.read_par_file('../../solver/INPUT/par_template.cfg')

g = np.fromfile('alpha_kernel.bin', dtype='float32')
g = g.reshape((p.nz, p.nx))

clim = 1e-6

gs = gridsmooth(g, 20)

plt.subplot(2,1,1)
plt.imshow(g)
plt.clim(-clim, clim)

plt.subplot(2,1,2)
plt.imshow(gs)
plt.clim(-clim, clim)

plt.show()
