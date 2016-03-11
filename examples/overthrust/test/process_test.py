import numpy as np
from obspy import read

d = read('Ux_data.su', format='SU', byteorder='<')
s = read('Ux_syn.su', format='SU', byteorder='<')

n = len(d)
freq = 2.5

for ir in range(n):
    d[ir].detrend()
    s[ir].detrend()
    d[ir].filter('lowpass', freq=2.5)
    s[ir].filter('lowpass', freq=2.5)
    s[ir].data = s[ir].data - d[ir].data
    s[ir].data = s[ir].data.astype(dtype=np.float32)

s.write('Res.su', format='SU', byteorder='<')
