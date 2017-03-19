
import sys
from os.path import join

from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import

from seisflows.plugins.encode import SourceArray, SourceGroups
from seisflows.plugins.solver.pewf2d import Par

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class ssewf2d(custom_import('solver', 'pewf2d')):
    """ Python interface for SSEWF2D
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(ssewf2d, self).check()

        # check paths
        if 'SOURCE' not in PATH:
            setattr(PATH, 'SOURCE', join(PATH.WORKDIR, 'sources'))

        # check pars
        if 'SOURCE_FILE' not in PAR:
            raise ParameterError(PAR, 'SOURCE_FILE')

        if 'NSOURCES' not in PAR:
            raise ParameterError(PAR, 'NSOURCES')

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', True)

        # check input
        if PAR.NSOURCES % PAR.NTASK != 0:
            raise NotImplementedError('Source groups must currently have equal size.')

    ### higher level interface

    def setup(self):
        """ Perform setup. Generates synthetic observed data.
        """
        # Initialize solver directories
        self.initialize_solver_directories()

    def setup_encoding(self):
        """ Setup encoding on head node.
        """
        # construct source array (all sources)
        self.source_array = SourceArray.fromfile(join(PATH.DATA, PAR.SOURCE_FILE))

        # check input
        if len(self.source_array) != PAR.NSOURCES:
            raise ValueError('NSOURCES in parameter file does not match length of input source file.')

        # create source groups (supershots)
        self.source_groups = SourceGroups()
        self.source_groups.group_sources(self.source_array, PAR.NTASK)

        if PAR.VERBOSE:
            self.source_array.print_positions()
            self.source_groups.print_groups()

    def generate_encoding(self):
        """ Generate encoding. Should be called on head node.
        """
        # generate encoding for current iteration
        self.source_groups.generate_encoding(reset=True)

        if PAR.VERBOSE:
            self.source_groups.print_groups()

    def generate_data(self):
        """ Perform shot encoding.
        """
        # get encoded shot path
        path = self.getpath
        itask = system.getnode()

        # get local source group information
        source_array = self.source_groups[itask]
        encoding = self.source_groups.encoding[itask]

        # save encoded source time functions and encoded data
        preprocess.encode_sources(path=path,
                                  itask=itask,
                                  source_array=source_array,
                                  encoding=encoding)

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
                         source_dir=PATH.SOURCE)

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
                         source_dir=PATH.SOURCE)

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
                         source_dir=PATH.SOURCE)
        self.forward()
