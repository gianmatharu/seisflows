import matplotlib.pyplot as plt
from fwpy.seistools.plotutils import Par
from obspy.segy.segy import readSU
import sys 
import numpy as np 

input_file = "../../../../solver/INPUT/par.cfg"

# Read parameter file
p = Par()
p.read_par_file(input_file)


for arg in sys.argv:
    print(arg)

fname = sys.argv[1]
traceid = int(sys.argv[2])

time = np.arange(0, p.ntimesteps * p.dt, p.dt)
d = readSU(fname, endian='<')

plt.plot(time, d.traces[traceid].data)
plt.show()

