
WORKFLOW='test_gradient'    # inversion
SOLVER='ewf2d'      # specfem2d, specfem3d
SYSTEM='serial'         # serial, pbs, slurm
OPTIMIZE='base'         # base, newton
PREPROCESS='ewf2d'       # base
POSTPROCESS='ewf2d'      # base

MISFIT='Waveform'
MATERIALS='Acoustic'
DENSITY = 'Constant'
CHANNELS = ['x', 'z']

# WORKFLOW
BEGIN=1
END=25
TITLE = 'OVERTHRUST'
WALLTIME = 60
MEMORY = 22

# PREPROCESSING
READER = 'su_ewf2d_obspy'             # seismic unix
WRITER = 'su_ewf2d_obspy'
NORMALIZE=False         # normalize traces
FREQHI=2.5
DAMPING = 2.0           # exponential time damping (per second)
GAIN=False

# POSTPROCESSING
SMOOTH=6.0

# OPTIMIZATION
SCHEME='LBFGS'
MASK=True
STEPMAX = 10

# SOLVER
LINE_DIR = 'x'             # receiver line direction (x or z)
LINE_START = 125.0        # First source position
DSRC = 200.0              # Source spacing
FIXED_POS = 50.0           # Fixed source spacing

# SYSTEM

NTASK = 1                # Number of shots
NPROC = 1
NODESIZE = 8
