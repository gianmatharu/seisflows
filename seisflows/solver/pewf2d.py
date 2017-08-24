
import sys
from os.path import join, basename
from glob import glob

import numpy as np

from seisflows.tools import unix
from seisflows.plugins import material
from seisflows.plugins.io.pewf2d import read, write
from seisflows.tools.tools import exists, call_solver
from seisflows.tools.array import gridsmooth, loadnpy
from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import  Par, read_cfg_file, write_cfg_file, event_dirname

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
preprocess = sys.modules['seisflows_preprocess']

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class pewf2d(object):
    """ Python interface for PEWF2D

      See base class for method descriptions.
      PEWF2D class differs in that the solver incorporates shot
      parallelism into the source code.
    """

    assert 'MATERIALS' in PAR
    assert 'DENSITY' in PAR

    parameters = []

    # parametrization definition & maps
    try:
        parClass = getattr(material, PAR.MATERIALS.lower())
    except:
        raise AttributeError('{} not found in module material'.format(PAR.MATERIALS))

    parameters = parClass.parameters
    par_map_forward = parClass.par_map_forward
    par_map_inverse = parClass.par_map_inverse

    if PAR.DENSITY == 'Variable':
        density_scaling = None
        parameters += ['rho']
    elif PAR.DENSITY == 'Scaling':
        # scale using gardener's relation
        density_scaling = True
    elif PAR.DENSITY == 'Constant':
        density_scaling = None

    def check(self):
        """ Checks parameters and paths
        """
        # check parameters
        if PAR.SYSTEM not in ['serial', 'westgrid']:
            raise ValueError('PEWF2D must be implemented with serial/westgrid system class.')

        if 'FORMAT' not in PAR:
            raise ParameterError(PAR, 'FORMAT')

        if PAR.FORMAT != 'su':
            raise ValueError('Format must be SU')

        if 'CLIP' not in PAR:
            setattr(PAR, 'CLIP', 0)

        if 'NPROC' not in PAR:
            raise ParameterError(PAR, 'NPROC')

        if 'RESCALE' not in PAR:
            setattr(PAR, 'RESCALE', False)

        # check scratch paths
        if 'SCRATCH' not in PATH:
            raise ParameterError(PATH, 'SCRATCH')

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        if 'SOLVER' not in PATH:
            if PATH.LOCAL:
                setattr(PATH, 'SOLVER', join(PATH.LOCAL, 'solver'))
            else:
                setattr(PATH, 'SOLVER', join(PATH.SCRATCH, 'solver'))

        # check solver input paths
        if 'SOLVER_BIN' not in PATH:
            raise ParameterError(PATH, 'SOLVER_BIN')

        if 'SOLVER_INPUT' not in PATH:
            raise ParameterError(PATH, 'SOLVER_INPUT')

        # assertions
        assert self.parameters != []

        # reparametrization (allows inverse mapping if needed)
        if PAR.MATERIALS not in ['Acoustic', 'Shear', 'Elastic']:
            self.reparam = True
        else:
            self.reparam = False

        if PAR.RESCALE:
            # self.scale holds a dict of rescaling values
            self.scale = material.ParRescaler.mean_scaling(self.load(PATH.MODEL_INIT)).scale
        else:
            self.scale = None

    def check_solver_parameters(self):
        """ Check solver specific parameters
        """
        if PAR.NTASK != p.ntask:
            raise ValueError('PAR.NTASK != nsrc in cfg')

    ### low level interface

    def forward(self):
        """ Perform forward simulation. Must launch from /bin.
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        call_solver(system.mpiexec(), script, PATH.WORKDIR + '/dump_fwd')

        unix.cd(PATH.WORKDIR)

    def adjoint(self):
        """ Perform adjoint simulation. Must launch from /bin
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        call_solver(system.mpiexec(), script, PATH.WORKDIR + '/dump_adj')

        unix.cd(PATH.WORKDIR)

    ### setup

    def initialize_solver_directories(self):
        """ Initialize solver directories.
        """
        unix.mkdir(self.getpath)
        unix.cd(self.getpath)

        # create directory structure
        unix.mkdir('traces/obs')
        unix.mkdir('traces/syn')
        unix.mkdir('traces/adj')

    ### higher level interface

    def setup(self):
        """ Perform setup. Generates synthetic observed data.
        """
        # Initialize solver directories
        self.check_solver_parameters()
        self.initialize_solver_directories()

        if PATH.DATA:
            # copy data to scratch dirs
            src = glob(PATH.DATA +'/'+ basename(self.getpath) +'/'+ '*')
            dst = 'traces/obs/'
            unix.cp(src, dst)

    def generate_data(self):
        """ Generate 'real' data in true model.
        """
        # generate data on the fly
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_TRUE,
                         output_dir=output_dir,
                         mode=0,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file=PAR.STF_FILE)

        self.forward()
        self.export_data()

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
                         stf_file='stf_f.txt')

        self.forward()

    def prepare_eval_grad(self):
        """ Prepare adjoint sources
        """
        preprocess.prepare_eval_grad(self.getpath)

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
                         stf_file='stf_f.txt')

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
                         stf_file='stf_f.txt')
        self.forward()

    def process_trial_step(self):
        """ Process line search data
        """
        itask = system.taskid()
        trial_dir = join(PATH.FUNC, event_dirname(itask + 1))

        # delete old file
        rfile = join(trial_dir, 'residuals')
        if exists(rfile):
            unix.rm(rfile)

        preprocess.evaluate_trial_step(self.getpath, trial_dir)

    def export_trial_solution(self, path=''):
        """ Save trial solution for frugal inversion.
        """
        # transfer synthetic data
        src = glob(join(path, basename(self.getpath), 'traces', 'syn', '*'))
        dst = join(self.getpath, 'traces', 'syn')
        unix.mv(src, dst)

   # serial/reduction function

    def combine(self, path='', parameters=[]):
        """ sum event gradients to compute misfit gradient
        """
        grad = {}

        # sum gradient
        for key in parameters or self.parameters:
            gradp = np.zeros(p.nx * p.nz, dtype='float32')
            for itask in range(PAR.NTASK):
                fpath = join(PATH.SOLVER, event_dirname(itask + 1), 'traces', 'syn')
                gradp += read(fpath, key, suffix='_kernel')

            grad[key] = gradp

            if PAR.RESCALE:
                grad[key] /= self.scale[key]

        # backup raw kernel
        self.save(grad, path, suffix='_kernel_raw')
        self.save(grad, path, suffix='_kernel')

    def smooth(self, path='', parameters=[], span=0.):
        """ Process gradient
        """
        grad = self.load(path, suffix='_kernel')

        for key in parameters or self.parameters:
            grad[key] = grad[key].reshape((p.nz, p.nx))
            grad[key] = gridsmooth(grad[key], span)

        self.save(grad, path, suffix='_kernel')

    # solver specific utils

    def export_data(self):
        """ Move data
        """
        for itask in range(PAR.NTASK):
            path = join(PATH.SOLVER, event_dirname(itask + 1))
            src = self.data_filenames(join(path, 'traces/syn'))
            dst = join(path, 'traces/obs')
            unix.mv(src, dst)

    def clean_boundary_storage(self):
        """ Delete boundary files required for wavefield reconstruction
        """
        for itask in range(PAR.NTASK):
            path = join(PATH.SOLVER, event_dirname(itask + 1), 'traces/syn')
            unix.rm(glob(join(path, 'proc*')))

    def load(self, path, prefix='', suffix='', rescale=False):
        """ Loads a model dictionary
        """
        model = {}
        for key in self.parameters:
            model[key] = read(path, key, prefix, suffix)

            if rescale:
                # rescale model files (warning, precond + masks use no prefix/suffix, ensure rescale=False)
                if not prefix and not suffix:
                    model[key] /= self.scale[key]
                elif suffix in ['_kernel']:
                    model[key] *= self.scale[key]

        return model

    def save(self, model, path, prefix='', suffix='', rescale=False):
        """ Saves model dictionary as solver binaries
        """
        # undo parameter scaling
        if rescale:
            if not prefix and not suffix:
                print('Fixing parameter scaling...')
                for key in self.parameters:
                    model[key] *= self.scale[key]

        # determine gradient call
        unix.mkdir(path)
        if suffix in ['_kernel']:
            grad_call = True
        else:
            grad_call = False

        # save inversion parameters
        for key in self.parameters:
            if PAR.SAFEUPDATE and not grad_call:
                model[key] = self.check_model(model[key], key)

            write(model[key], path, key, prefix, suffix)

        # skip reparametrization for kernels
        if grad_call:
            return

        # if using a non-velocity parametrization
        if self.reparam:
            # handle density
            if 'rho' not in self.parameters:
                # load current density model
                model['rho'] = read(PATH.MODEL_EST, 'rho')

            # overwrite model
            model = self.par_map_inverse(model)

        # Assigning solver parameters
        for key in ['vp', 'vs']:
            if key not in model.keys():
                src = join(PATH.MODEL_INIT, key + '.bin')
                dst = path
                unix.cp(src, dst)
            else:
                write(model[key], path, key, prefix, suffix)

        if 'rho' not in self.parameters:
            if self.density_scaling:
                rho = getattr(material, 'gardeners')(model)
                write(rho, path, 'rho', prefix, suffix)
            else:
                # copy density from initial model
                src = join(PATH.MODEL_INIT, 'rho.bin')
                dst = join(path, 'rho.bin')
                unix.cp(src, dst)

    def merge(self, model):
        """ Merge vectors, used to merge model/gradient binary files into a single vector
        """
        v = np.array([])
        for key in self.parameters:
            v = np.append(v, model[key])

        return v

    def split(self, v):
        """ Split numpy arrays into separate vectors and save to binary files for solver.
        """
        npar = len(self.parameters)
        n = int(len(v) / npar)
        model = {}

        for ipar, key in enumerate(self.parameters):
            vpar = v[(ipar*n):(ipar*n) + n]
            model[key] = vpar

        return model

    # solver specific routines
    def check_model(self, v, par):
        """ Perform bounds check on model update
        """
        # get user prescribed limits
        minval = getattr(PAR, str(par + 'min').upper())
        maxval = getattr(PAR, str(par + 'max').upper())

        if minval >= maxval:
            raise ValueError('{} min val greater than max val!'.format(par))

        if minval < 0:
            minval = 0

        if maxval <= 0:
            raise ValueError('{} max val greater must be > 0!'.format(par))

        # check min values
        indlow = v < minval
        v[indlow] = minval

        # check max values
        indhigh = v > maxval
        v[indhigh] = maxval

        return v

    # configuration file handling

    def set_par_cfg(self, **kwargs):
        """ Sets parameter cfg file for solver. Keys must match solver par.cfg file parameters.
        """
        # read par dict
        cfg = read_cfg_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))

        for key, value in kwargs.items():
            if key in cfg:
                # adjust parameters
                if isinstance(value, basestring):
                    cfg[key] = '"' + value + '"'
                elif isinstance(value, bool):
                    cfg[key] = str(value).lower()
                else:
                    cfg[key] = value
            else:
                raise KeyError

        # write par dict
        write_cfg_file(join(PATH.SOLVER_INPUT, 'par.cfg'), cfg)

    @property
    def getpath(self):
        itask = system.taskid()
        return join(PATH.SOLVER, event_dirname(itask + 1))

    def data_filenames(self, path):

        filenames = []
        filenames += glob(join(path, '*.su'))
        filenames += glob(join(path, '*.txt'))
        filenames += glob(join(path, '*.bin'))

        return filenames
