from os.path import join, basename

import numpy as np
from glob import glob
import subprocess

from seisflows.seistools.ewf2d import Par, read_cfg_file, write_cfg_file, get_cfg_value, event_dirname
from seisflows.tools import unix
from seisflows.tools.code import exists, mpicall
from seisflows.tools.array import gridsmooth, loadnpy, savenpy
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import preprocess

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))

class pewf2d(custom_import('solver', 'base')):

    def check(self):
        super(pewf2d, self).check()

        # check parameters

        if PAR.SYSTEM != 'serial':
            raise ValueError('Must be implemented with serial system class.')

        if 'FORMAT' not in PAR:
            raise ParameterError(PAR, 'FORMAT')

        if PAR.FORMAT != 'su':
            raise ValueError('Format must be SU')

        if PAR.NTASK != p.ntask:
            raise ValueError('PAR.NTASK != nsrc in cfg')

    ### low level interface

    def forward(self):
        """ Perform forward simulation. Must launch from /bin.
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        mpicall(system.mpiexec(), script, PATH.SUBMIT + '/dump')

        unix.cd(PATH.SUBMIT)

    def adjoint(self):
        """ Perform adjoint simulation. Must launch from /bin
        """
        unix.cd(PATH.SOLVER_BIN)
        script = './xewf2d'
        mpicall(system.mpiexec(), script, PATH.SUBMIT + '/dump')

        unix.cd(PATH.SUBMIT)

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
        self.initialize_solver_directories()

        if PATH.DATA:
            # copy data to scratch dirs
            src = glob(PATH.DATA +'/'+ basename(self.getpath) +'/'+ '*')
            dst = 'traces/obs/'
            unix.cp(src, dst)

    def generate_data(self):

        # generate data on the fly
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_TRUE,
                         output_dir=output_dir,
                         mode=0,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file=PAR.STF_FILE)

        self.forward()
        self.organize_output()

    def generate_synthetics(self, mode=0):

        if mode != 0 and mode != 1:
            raise ValueError('Mode must be set to forward (0) or save wavefield mode (1)')

        # generate synthetic data
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_INIT,
                         output_dir=output_dir,
                         mode=mode,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file=PAR.STF_FILE)

        self.forward()

    def prepare_eval_grad(self):

        # prepare adjoint sources
        preprocess.prepare_eval_grad(self.getpath)

    def compute_gradient(self):

        # generate synthetic data
        output_dir = join(PATH.SOLVER)
        self.set_par_cfg(external_model_dir=PATH.MODEL_INIT,
                         output_dir=output_dir,
                         adjoint_dir='traces/adj',
                         mode=2,
                         use_stf_file=PAR.USE_STF_FILE,
                         stf_file=PAR.STF_FILE)

        self.adjoint()
        self.clean_boundary_storage()

   # serial/reduction function

    def evaluate_gradient(self, model_dir=''):
        """ Compute event gradient by running adjoint simulation
        """

        # get task number
        itask = system.getnode()

        # setup directories
        syn_dir = join(self.getpath, 'traces', 'syn')
        adj_dir = join(self.getpath, 'traces', 'adj')

        # set par.cfg file for solver
        self.set_par_cfg(external_model_dir=model_dir, output_dir=syn_dir, save_final_wavefield=False,
                         adjoint_sim=True, adjoint_dir=adj_dir)

        # set src.cfg for solver
        xsrc = self.sources[itask][0]
        zsrc = self.sources[itask][1]
        self.set_src_cfg(xs=float(xsrc), zs=float(zsrc), use_stf_file=True, stf_file='stf_f.txt')

        # copy cfg files
        unix.cp(join(self.getpath, 'INPUT', 'par.cfg'), adj_dir)
        unix.cp(join(self.getpath, 'INPUT', 'src.cfg'), adj_dir)

        # run adjoint sim
        self.adjoint()

        # clean saved boundaries
        unix.rm(glob(join(syn_dir, 'proc*')))

    def combine(self, precond=True):
        """ sum event gradients to compute misfit gradient
        """

        # sum gradient
        for par in self.parameters:
            filename = par + '_kernel.bin'
            gradient = np.zeros((p.nz * p.nx), dtype='float32')

            for itask in range(PAR.NTASK):
                syn_dir = join(PATH.SOLVER, event_dirname(itask + 1), 'traces', 'syn')

                ev_grad = self.load(join(syn_dir, filename))

                if precond:
                    ev_grad = self.apply_preconditioner(syn_dir, ev_grad)

                gradient += ev_grad
                self.save(join(PATH.GRAD, filename), gradient)

    def smooth(self, span=0.):
        """ Process gradient
        """

        for par in self.parameters:
            filename = par + '_kernel.bin'
            g = self.load(join(PATH.GRAD, filename))
            g = g.reshape((p.nz, p.nx))
            g = gridsmooth(g, span)
            self.save(join(PATH.GRAD, par + '_smooth_kernel.bin'), g)

        g_new = self.merge(PATH.GRAD, '_smooth_kernel.bin')
        savenpy(join(PATH.OPTIMIZE, 'g_new'), g_new)

    # solver specific utilities

    def organize_output(self):

        for itask in range(PAR.NTASK):
            path = join(PATH.SOLVER, event_dirname(itask + 1))
            src = self.data_filenames(join(path, 'traces/syn'))
            dst = join(path, 'traces/obs')
            unix.mv(src, dst)

    def clean_boundary_storage(self):
        for itask in range(PAR.NTASK):
            path = join(PATH.SOLVER, event_dirname(itask + 1), 'traces/syn')
            unix.rm(glob(join(path, 'proc*')))

    def load(self, filename):
        """ Loads a float32 numpy 2D array
        """

        try:
            arr = np.fromfile(filename, dtype='float32')
        except:
            raise IOError('Could not read file: {}'.format(filename))

        return arr

    def save(self, filename, arr):
        """ Saves a numpy float32 array as a binary
        """
        arr.astype('float32').tofile(filename)

    def merge(self, path, suffix):
        """ Merge vectors, used to merge model/gradient binary files into a single vector
        """

        v = np.array([])
        for par in self.parameters:
            filename = join(path, par + suffix)
            vpar = self.load(filename)
            v = np.append(v, vpar)

        return v

    def split(self, file, path, suffix):
        """ Split numpy arrays into separate vectors and save to binary files for solver.
        """

        npar = len(self.parameters)
        ipar = 0

        # load numpy file
        nv = loadnpy(file)
        n = int(len(nv) / npar)

        for par in self.parameters:
            v = nv[(ipar*n):(ipar*n) + n]
            filename = join(path, par + suffix)
            if PAR.SAFEUPDATE:
                self.check_velocity_model(v, par)

            self.save(filename, v)
            ipar += 1

    # solver specific routines
    def check_velocity_model(self, v, par):

        # check min values
        minval = PAR.__getattr__(str(par + 'min').upper())
        maxval = PAR.__getattr__(str(par + 'max').upper())

        indlow = v < minval
        v[indlow] = minval

        # check max values
        indhigh = v > maxval
        v[indhigh] = maxval

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
        itask = system.getnode()
        return join(PATH.SOLVER, event_dirname(itask + 1))

    def data_filenames(self, path):

        filenames = []
        filenames += glob(join(path, '*.su'))
        filenames += glob(join(path, '*.txt'))
        filenames += glob(join(path, '*.bin'))

        return filenames
