
WORKFLOW='frugal_inversion'    # inversio
SOLVER='pewf2d'      # specfem2d, specfem3d
SYSTEM='serial'         # serial, pbs, slurm
OPTIMIZE='LBFGS'         # base, newton
PREPROCESS='pewf2d'       # base
POSTPROCESS='pewf2d'      # base

MISFIT='Waveform'
MATERIALS='Elastic'
DENSITY='Variable'
CHANNELS = ['x', 'z']

TEST='SPIKE'
PERC_PERT = None
SPIKE_X = [20, 200, 30]
SPIKE_Z = [20, 100, 30]

SAFEUPDATE = False  # Safe update flag imposes a bounds constraint on the
VPMIN = 2000.0       # model. If set to true, updates beyond the thresholds
VPMAX = 3500.0       # will be replaced with the boundary values.
VSMIN = 1500.0
VSMAX = 2000.0
RHOMIN = 750
RHOMAX = 2000
PPMIN = 1 / 3300.0
PPMAX = 1 / 2900.0
PSMIN = 1 / 2000.0
PSMAX = 1 / 1600.0

IPMIN = VPMIN * RHOMIN
IPMAX = VPMAX * RHOMAX
ISMIN = VSMIN * RHOMIN
ISMAX = VSMAX * RHOMAX

# WORKFLOW
BEGIN=1
END=50
SAVETRACES = 1
SAVERESIDUALS = 1

# PREPROCESSING
FORMAT = 'su'
READER = 'su_pewf2d'
WRITE = 'su_pewf2d'
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
MAX_OFFSET = 0.5        # max offset (km) for offset mute.
INNER_MUTE = False
#FREQHI=5.5             # corner frequency for filtering

# POSTPROCESSING
SMOOTH=3.0
CLIP = 10

# OPTIMIZATION
#PRECOND='diagonal'
LBFGSMEM = 8
STEPMAX=15
STEPINIT=0.005

# SYSTEM
NTASK = 16               # Number of shots
NPROC = 8               # Number of processes per simulation
