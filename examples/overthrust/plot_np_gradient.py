from fwpy.seistools.plotutils import Par
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/gmatharu/FwPy/examples/kevmousi')
import parameters

input_file = "../../solver/INPUT/par.cfg"

# Read parameter file
p = Par()
p.read_par_file(input_file)


npar = len(parameters.PARAMETERS)
print(npar)

# Read input file
args =[]

for arg in sys.argv:
    print(arg)

fname = sys.argv[1]
print('Plotting file ', fname)

# split vector
ipar = 0

# load numpy file
nv = np.load(fname)
n = len(nv) / npar

for par in parameters.PARAMETERS:
    v = nv[(ipar*n):(ipar*n) + n]
    ipar += 1

npad = p.npad
nx = p.nx
nz = p.nz

lim = sys.argv[2]
clim = abs(max(v)/10)
v = v.reshape((nz, nx))

plt.imshow(v)
plt.title('Vp kernel')
if int(lim) == 1:
    plt.clim(-clim, clim)

plt.show()
