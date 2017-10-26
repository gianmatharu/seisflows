
import sys
from os.path import join
from glob import glob

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.tools import saveobj, divides, exists
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import iter_dirname, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']


class saga_inversion(custom_import('workflow', 'stochastic_inversion')):
    """ SAGA workflow inversion class. 

      Compute iterative non-linear inversion. Designed to fit PEWF2D solver.
      Follows a more generic design to base class.
      
      Requires an initial gradient for all sources. 
    """

    def check(self):
        """ Check parameters and paths
        """
        # Inherit from grandparent (p_inversion) class
        super(custom_import('workflow', 'stochastic_inversion'), self).check()

        # check paths
        if not PATH.DATA:
            raise ValueError('Data path required.')

        if 'SAGAINIT' not in PATH:
            raise ValueError('Requires first iteration gradient as input')

        # Path for SAGA updates
        if 'SAGA' not in PATH:
            setattr(PATH, 'SAGA', join(PATH.WORKDIR, 'SAGA'))

        # check parameters
        if PAR.PREPROCESS != 'pewf2d':
            raise ValueError('Can only run stochastic inversion with pewf2d preprocessing class.')

        if PAR.SOLVER != 'saga_pewf2d':
            raise ValueError('Can only run stochastic inversion with spewf2d solver class.')

        if 'SAVESUBSET' not in PAR:
            setattr(PAR, 'SAVESUBSET', 1)

        if PAR.OPTIMIZE != 'steepest_descent':
            raise ValueError('Class must use SD optimization')

    def setup(self):
        """ Lays groundwork for inversion
        """
        super(saga_inversion, self).setup()
        # clean scratch directories
        if PAR.BEGIN == 1:
            # copy initial set of gradients
            unix.rm(PATH.SAGA)
            unix.cp(PATH.SAGAINIT, PATH.SAGA)


    def compute_gradient(self):
        """ compute aggregate gradient
        """
        super(saga_inversion, self).compute_gradient()
        system.run_single('solver', 'update_aggregate_gradient', path=PATH.SAGA)