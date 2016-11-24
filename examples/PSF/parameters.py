
WORKFLOW='p_inversion'    # inversio
SOLVER='pewf2d'      # specfem2d, specfem3d
SYSTEM='serial'         # serial, pbs, slurm
OPTIMIZE='base'         # base, newton
PREPROCESS='ewf2d'       # base
POSTPROCESS='ewf2d'      # base

MISFIT='Waveform'
MATERIALS='Elastic'
DENSITY='Constant'
CHANNELS = ['x', 'z']

SAFEUPDATE = True  # Safe update flag imposes a bounds constraint on the
VPMIN = 2500.0       # model. If set to true, updates beyond the thresholds
VPMAX = 3500.0       # will be replaced with the boundary values.
VSMIN = 1500.0
VSMAX = 2000.0
RHOMIN = 750
RHOMAX = 2000

# WORKFLOW
BEGIN=1
END=50
SAVETRACES = 1
SAVERESIDUALS = 1

# PREPROCESSING
FORMAT = 'su'
READER = 'su_ewf2d_obspy'
WRITE = 'su_ewf2d_obspy'
USE_STF_FILE = True
STF_FILE = 'stf.txt'

NORMALIZE=False         # normalize traces
#GAIN=False              # apply offset dependent gain
#DAMPING = 1.5           # exponential time damping (per second)
MUTE_WINDOW = False      # mute outside of a time window
WINDOW = 'tukey'        # taper functions - tukey (default), cosine, gaussian
TMIN = 0.5              # start of time window
TMAX = 5.0              # end of time window
MUTE_OFFSET = False
MAX_OFFSET = 2.0        # max offset (km) for offset mute.
INNER_MUTE = False
#FREQHI=5.5             # corner frequency for filtering

# POSTPROCESSING
SMOOTH=6.0
#CLIP = 4

# OPTIMIZATION
#PRECOND='pewf2d_diagonal'
SCHEME='LBFGS'
LBFGSMEM = 8
STEPMAX=15
STEPINIT=0.05

# SOLVER
USE_SRC_FILE=False
LINE_DIR = 'x'             # receiver line direction (x or z)
LINE_START = 325.0        # First source position
DSRC = 400.0              # Source spacing
FIXED_POS = 50.0           # Fixed source spacing

# SYSTEM

NTASK = 16               # Number of shots
NPROC = 8               # Number of processes per simulation
NPROCMAX = 8
