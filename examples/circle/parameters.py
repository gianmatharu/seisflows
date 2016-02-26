
WORKFLOW='test_gradient'    # inversion
SOLVER='ewf2d'      # specfem2d, specfem3d
SYSTEM='serial'         # serial, pbs, slurm
OPTIMIZE='base'         # base, newton
PREPROCESS='ewf2d'       # base
POSTPROCESS='ewf2d'      # base

MISFIT='Waveform'
MATERIALS='Acoustic'
DENSITY='Constant'
CHANNELS = ['x', 'z']

# WORKFLOW
BEGIN=1
END=25
#SAVEGRADIENT=1
#SAVETRACES=0

# PREPROCESSING
READER = 'su_ewf2d_obspy'
WRITE = 'su_ewf2d_obspy'
NORMALIZE=False         # normalize traces
#DAMPING = 0.05           # exponential time damping (per second)
FREQHI=3.0
#GAIN=True

# POSTPROCESSING
SMOOTH=7

# OPTIMIZATION
SCHEME='LBFGS'
MASK=True               # Applies preconditioner
STEPMAX=5

# SOLVER
USE_SRC_FILE=True
LINE_DIR = 'x'             # receiver line direction (x or z)
LINE_START = 200.0        # First source position
DSRC = 200.0              # Source spacing
FIXED_POS = 50.0           # Fixed source spacing

# SYSTEM

NTASK = 4                 # Number of shots
NPROC = 1               # Number of processes per simulation

