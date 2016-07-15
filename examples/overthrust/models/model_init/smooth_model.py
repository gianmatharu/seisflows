import sys
import numpy as np
from seisflows.tools.array import readgrid, gridsmooth

sigma = sys.argv[1]
sigma = float(sigma)

nx = 800
nz = 180

vp = readgrid('vp.bin', nx, nz, dtype='float32')
vs = readgrid('vs.bin', nx, nz, dtype='float32')

vp = gridsmooth(vp, sigma)
vs = gridsmooth(vs, sigma)

vp.astype('float32').tofile('vps.bin')
vs.astype('float32').tofile('vss.bin')
