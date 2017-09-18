
import sys
from os.path import join, exists
from glob import glob

import numpy as np

from seisflows.tools import unix
from seisflows.plugins.solver.pewf2d import Par, event_dirname
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.encode import SourceArray, decimate_source_array

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class spewf2d(custom_import('solver', 'pewf2d')):
    """ Python interface for PEWF2D. Specialized class for 
        stochastic inversion

      See base class for method descriptions.
      PEWF2D class differs in that the solver incorporates shot
      parallelism into the source code.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(spewf2d, self).check()

        # check parameters
        if 'SOURCE_FILE' not in PAR:
            raise ParameterError(PAR, 'SOURCE_FILE')

        if 'NSOURCES' not in PAR:
            raise ParameterError(PAR, 'NSOURCES')

        if 'STOCHASTIC' not in PAR:
            setattr(PAR, 'STOCHASTIC', False)

        if 'ITER_RESET' not in PAR:
            setattr(PAR, 'ITER_RESET', 1)

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', False)

        # check paths
        if 'SOURCE' not in PATH:
            setattr(PATH, 'SOURCE', join(PATH.WORKDIR, 'sources'))

   ### higher level interface

    def setup(self):
        """ Perform setup. Generates synthetic observed data.
        """
        # Initialize solver directories
        self.initialize_solver_directories()

        if not exists(PATH.SOURCE):
            unix.mkdir(PATH.SOURCE)

    def setup_sources(self):
        """ Select samples for stochastic optimization
        """
        # construct source array (all sources)
        self.source_array = SourceArray.fromfile(join(PATH.DATA, PAR.SOURCE_FILE))

        # check input
        if len(self.source_array) != PAR.NSOURCES:
            raise ValueError('NSOURCES in parameter file does not match length of input source file.')

        if PAR.VERBOSE:
            self.source_array.print_positions()

    def select_sources(self):
        """ Decimate source array.
        """
        # generate working source array
        self.source_array_subset = decimate_source_array(self.source_array, PAR.NTASK, random=PAR.STOCHASTIC)
        self._write_source_file()

        if PAR.VERBOSE:
            print 'Current subset...'
            self.source_array_subset.print_positions()


    def fetch_data(self):
        """ Copy data to working directory. 
        """
        for itask in xrange(PAR.NTASK):
            ishot = self.source_array_subset[itask].index
            src = glob(PATH.DATA +'/'+ event_dirname(ishot) +'/'+ '*')
            dst = join(PATH.SOLVER, event_dirname(itask+1), 'traces/obs/')
            unix.cp(src, dst)


    def generate_synthetics(self, mode=0):
        """ Generate synthetic data in estimated model.
        """
        if not (mode == 0 or mode == 1):
            raise ValueError('Mode must be set to forward (0) or save wavefield mode (1)')

        # generate synthetic data
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_EST,
                         output_dir=output_dir,
                         mode=mode,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file='stf_f.txt',
                         use_src_file=True,
                         src_file=join(PATH.SOURCE, 'SOURCES'))

        self.forward()

    def compute_gradient(self):
        """ Compute gradient
        """
        # generate synthetic data
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_EST,
                         output_dir=output_dir,
                         adjoint_dir='traces/adj',
                         mode=2,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file='stf_f.txt',
                         use_src_file=True,
                         src_file=join(PATH.SOURCE, 'SOURCES'))

        self.adjoint()
        self.clean_boundary_storage()

    def evaluate_function(self, mode=0):
        """ Evaluate test function for a trial model
        """
        if not (mode == 0 or mode == 1):
            raise ValueError('Mode must be set to forward (0) or save wavefield mode (1)')

        output_dir = join(PATH.FUNC)
        model_dir = join(PATH.FUNC, 'model')
        unix.mkdir(output_dir)

        self.set_par_cfg(external_model_dir=model_dir,
                         output_dir=output_dir,
                         mode=mode,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file='stf_f.txt',
                         use_src_file=True,
                         src_file=join(PATH.SOURCE, 'SOURCES'))
        self.forward()

    def _write_source_file(self):
        """ Write solver suitable source file.
        """
        filename = join(PATH.SOURCE, 'SOURCES')
        with open(filename, 'w') as f:
            for i in xrange(len(self.source_array_subset)):
                f.write('{:.6f} {:.6f}\n'.format(self.source_array_subset[i].x, self.source_array_subset[i].z))

        return