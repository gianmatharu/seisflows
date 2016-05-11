
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

# WORKFLOW
BEGIN=1
END=25

# PREPROCESSING
READER = 'su_ewf2d_obspy'
WRITE = 'su_ewf2d_obspy'
NORMALIZE=False         # normalize traces
DAMPING = 0.5           # exponential time damping (per second)
FREQHI=12.0
#GAIN=True

# POSTPROCESSING
SMOOTH=3.0

# OPTIMIZATION
SCHEME='NLCG'
MASK=True   # Applies preconditione
STEPMAX=5
PRECOND_TYPE='IGEL'

# SOLVER
USE_SRC_FILE=False
LINE_DIR = 'x'             # receiver line direction (x or z)
LINE_START = 280.0        # First source position
DSRC = 100.0              # Source spacing
FIXED_POS = 40.0           # Fixed source spacing

# SYSTEM

NTASK = 96               # Number of shots
NPROC = 1               # Number of processes per simulation
NPROCMAX= 8
