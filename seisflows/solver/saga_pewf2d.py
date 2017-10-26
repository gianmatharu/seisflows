
import sys
from os.path import join, exists
from glob import glob

import numpy as np

from seisflows.tools import unix
from seisflows.config import custom_import
from seisflows.plugins.solver_io.pewf2d import read
from seisflows.plugins.encode import SourceArray
from seisflows.plugins.solver.pewf2d import Par, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class saga_pewf2d(custom_import('solver', 'spewf2d')):
    """ Python interface for PEWF2D. Specialized class for 
        stochastic inversion (SAGA)

      See base class for method descriptions.
      PEWF2D class differs in that the solver incorporates shot
      parallelism into the source code.
    """
    def combine(self, path='', parameters=[]):
        """ sum event gradients to compute misfit gradient
        """
        grad = {}
        subset_remain = SourceArray(set(self.source_array) - set(self.source_array_subset))
        subset_remain.print_positions()

        # sum gradient
        for key in parameters or self.parameters:
            gradp = np.zeros(p.nx * p.nz, dtype='float32')

            # sum contributions from subset
            for itask, source in enumerate(self.source_array_subset):
                fpath = join(PATH.SOLVER, event_dirname(itask+1), 'traces/syn')
                gradp += read(fpath, key, suffix='_kernel')

            for source in subset_remain:
                fpath = join(PATH.SAGA, event_dirname(source.index))
                gradp += read(fpath, key, suffix='_kernel')

            grad[key] = gradp
            if PAR.RESCALE:
                grad[key] *= self.scale[key]

        # backup raw kernel
        self.save(grad, path, suffix='_kernel')

    def update_aggregate_gradient(self, path='', parameters=[]):
        """ Update aggregate gradient estimate with latest iteration gradients.  
        """
        for itask, source in enumerate(self.source_array_subset):

            kernel_path = join(PATH.SOLVER, event_dirname(itask + 1), 'traces/syn')
            src = [join(kernel_path, item + '_kernel.bin')
                            for item in parameters or self.parameters]
            dst = join(path, event_dirname(source.index))
            unix.cp(src, dst)

