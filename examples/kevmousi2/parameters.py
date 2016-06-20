
WORKFLOW='test_gradient'    # inversion
SOLVER='ewf2d'      # specfem2d, specfem3d
SYSTEM='mpi_queue'         # serial, pbs, slurm
OPTIMIZE='base'         # base, newton
PREPROCESS='ewf2d'       # base
POSTPROCESS='ewf2d'      # base

MISFIT='Waveform'
MATERIALS='Elastic'
DENSITY='Constant'
CHANNELS = ['x', 'z']

SAFEUPDATE = True  # Safe update flag imposes a bounds constraint on the
VPMIN = 1580.0       # model. If set to true, updates beyond the thresholds
VPMAX = 3000.0       # will be replaced with the boundary values.
VSMIN = 897.0
VSMAX = 1704.0

# WORKFLOW
BEGIN=1
END=50
SAVETRACES = 1
SAVERESIDUALS = 1

# PREPROCESSING
READER = 'su_ewf2d_obspy'
WRITE = 'su_ewf2d_obspy'
USE_STF_FILE = True
STF_FILE = 'stf.txt'

NORMALIZE=False         # normalize traces
#GAIN=False              # apply offset dependent gain
#DAMPING = 2.0           # exponential time damping (per second)
MUTE_WINDOW = False   # mute outside of a time window
WINDOW = 'tukey'        # taper functions - tukey (default), cosine, gaussian
TMIN = 0.5              # start of time window
TMAX = 6.0              # end of time window
MUTE_OFFSET = False
MAX_OFFSET = 2.0        # max offset (km) for offset mute.
INNER_MUTE = True
#FREQHI=3.0             # corner frequency for filtering

# POSTPROCESSING
SMOOTH=4.0

# OPTIMIZATION
SCHEME='NLCG'
#LBFGSMEM = 8
MASK=True  # Applies precondition
STEPMAX=10
STEPINIT=0.05
PRECOND_TYPE='ONE_WAY'
PRECOND_THRESH = 1e-3
PRECOND_SMOOTH = 10

# SOLVER
USE_SRC_FILE=False
LINE_DIR = 'x'             # receiver line direction (x or z)
LINE_START = 260.0        # First source position
#DSRC = 100.0              # Source spacing
DSRC = 200.0
FIXED_POS = 40.0           # Fixed source spacing

# SYSTEM

#NTASK = 96               # Number of shots
NTASK = 48
NPROC = 1               # Number of processes per simulation
NPROCMAX= 8
