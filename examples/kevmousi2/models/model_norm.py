import sys
import numpy as np
from os.path import join
from seisflows.tools.array import readgrid


def norm(mk, mt, n):
    mdiff = (mk - mt) / mt
    return np.linalg.norm(mdiff) / n

path = sys.argv[1]
maxiter = int(sys.argv[2])

nx = 500
nz = 130
n = nx * nz
mnorm = np.zeros(maxiter+1)

mtpath = '/home/gian/Desktop/seisflows/examples/kevmousi2/models/model_true'
mt = readgrid(join(mtpath, 'vp.bin'), nx, nz, dtype='float32')

mipath = join('.', path, 'model_init')
mi = readgrid(join(mipath, 'vp.bin'), nx, nz, dtype='float32')

mnorm[0] = norm(mi, mt, n)

for it in range(maxiter):

    mpath = join('.', path, 'm{:02d}'.format(it + 1))
    m = readgrid(join(mpath, 'vp.bin'), nx, nz, dtype='float32')
    mnorm[it + 1] = norm(m, mt, n)

for it in range(maxiter+1):
    print '{:.3e}'.format(mnorm[it])

