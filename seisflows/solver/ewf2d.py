from os.path import join, basename

import numpy as np
from glob import glob
import subprocess
import os

from seisflows.seistools import misfit
from seisflows.seistools.ewf2d import Par, read_cfg_file, write_cfg_file, event_dirname, extend_pml_velocities

from seisflows.tools import unix
from seisflows.tools.code import exists
from seisflows.tools.array import gridsmooth, loadnpy, savenpy
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, ParameterError, custom_import

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()

import system
import preprocess

# Solver parameter class
p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class ewf2d(custom_import('solver', 'base')):

    def check(self):
        super(ewf2d, self).check()

        # check parameters
        if 'LINE_DIR' not in PAR:
            setattr(PAR, 'LINE_DIR', 'x')

        if 'LINE_START' not in PAR:
            raise ParameterError(PAR, 'LINE_START')

        if 'DSRC' not in PAR:
            raise ParameterError(PAR, 'DSRC')

        if 'FIXED_POS' not in PAR:
            raise ParameterError(PAR, 'FIXED_POS')

        if 'USE_SRC_FILE' not in PAR:
            setattr(PAR, 'USE_SRC_FILE', False)

        if 'LS_NODE' not in PAR:
            setattr(PAR, 'LS_NODE', 'getnode')

        self.set_source_array()

    ### low level interface

    def forward(self):
        """ Perform forward simulation. Must launch from /bin.
        """
        unix.cd(join(self.getpath, 'bin'))
        script = './xewf2d'
        super(ewf2d, self).mpirun(script, PATH.SUBMIT + '/dump')

        unix.cd(PATH.SUBMIT)

    def adjoint(self):
        """ Perform adjoint simulation. Must launch from /bin
        """
        unix.cd(join(self.getpath, 'bin'))
        script = './xewf2d'
        super(ewf2d, self).mpirun(script, PATH.SUBMIT + '/dump')

        unix.cd(PATH.SUBMIT)

    ### setup

    def initialize_solver_directories(self):
        """ Initialize solver directories.
        """
        unix.mkdir(self.getpath)
        unix.cd(self.getpath)

        # create directory structure
        unix.mkdir('INPUT')
        unix.mkdir('bin')

        unix.mkdir('traces/obs')
        unix.mkdir('traces/syn')
        unix.mkdir('traces/adj')

        # copy exectuables
        src = glob(join(PATH.SOLVER_BIN, '*'))
        dst = 'bin/'
        unix.cp(src, dst)

        # copy input files
        src = glob(join(PATH.SOLVER_INPUT, '*'))
        dst = 'INPUT/'
        unix.cp(src, dst)

    ### higher level interface

    def setup(self):
        """ Perform setup. Generates synthetic observed data.
        """

        # clean up solver directories
        unix.rm(self.getpath)
        self.initialize_solver_directories()

        if PATH.DATA:
            # copy data to scratch dirs
            src = glob(PATH.DATA +'/'+ basename(self.getpath) +'/'+ '*')
            dst = 'traces/obs/'
            unix.cp(src, dst)

        else:
            # generate data on the fly
            output_dir = join(self.getpath, 'traces', 'obs')
            self.generate_data(model_dir=PATH.MODEL_TRUE, output_dir=output_dir)

    def compute_gradient(self):
        """ Sequential event gradient computation, reduces storage requirement.
        """

        # generate synthetic data
        output_dir = join(self.getpath, 'traces', 'syn')
        self.generate_data(model_dir=PATH.MODEL_EST, output_dir=output_dir, save_wavefield=True)

        # prepare adjoint sources
        preprocess.prepare_eval_grad(self.getpath)

        # compute event gradient
        self.evaluate_gradient(model_dir=PATH.MODEL_EST)

    def evaluate_function(self, path=''):
        """ Evaluate test function
        """
        # generate synthetic data
        itask = getattr(system, PAR.LS_NODE)()

        output_dir = join(path, event_dirname(itask + 1))
        model_dir = join(path, 'model')
        unix.mkdir(output_dir)

        self.generate_data(model_dir=model_dir, output_dir=output_dir)

        preprocess.evaluate_trial_step(self.getpath, output_dir)

    # per event functions

    def generate_data(self, model_dir=PATH.MODEL_TRUE, output_dir='', save_wavefield=False):
        """ Generate dataset. Defaults to generating synthetic data for true model.
        """

        # get task number
        itask = system.getnode()

        # set par.cfg file for solver
        self.set_par_cfg(external_model_dir=model_dir, output_dir=output_dir, save_forward_wavefield=save_wavefield)

        # set src.cfg for solver
        xsrc = self.sources[itask][0]
        zsrc = self.sources[itask][1]
        self.set_src_cfg(xs=float(xsrc), zs=float(zsrc))

        # copy cfg files
        unix.cp(join(self.getpath, 'INPUT', 'par.cfg'), output_dir)
        unix.cp(join(self.getpath, 'INPUT', 'src.cfg'), output_dir)

        # run forward sim
        self.forward()

    def evaluate_gradient(self, model_dir=''):
        """ Compute event gradient by running adjoint simulation
        """

        # get task number
        itask = system.getnode()

        # setup directories
        syn_dir = join(self.getpath, 'traces', 'syn')
        adj_dir = join(self.getpath, 'traces', 'adj')

        # set par.cfg file for solver
        self.set_par_cfg(external_model_dir=model_dir, output_dir=syn_dir, save_forward_wavefield=False,
                         adjoint_sim=True, adjoint_dir=adj_dir)

        # set src.cfg for solver
        xsrc = self.sources[itask][0]
        zsrc = self.sources[itask][1]
        self.set_src_cfg(xs=float(xsrc), zs=float(zsrc))

        # copy cfg files
        unix.cp(join(self.getpath, 'INPUT', 'par.cfg'), adj_dir)
        unix.cp(join(self.getpath, 'INPUT', 'src.cfg'), adj_dir)

        # run adjoint sim
        self.adjoint()

        # clean saved boundaries
        unix.rm(glob(join(syn_dir, 'proc*')))

    # serial/reduction function

    def combine(self):
        """ sum event gradients to compute misfit gradient
        """

        # sum gradient preconditioner
        precond = np.zeros((p.nz * p.nx), dtype='float32')
        filename = 'precond.bin'
        for itask in range(PAR.NTASK):
            syn_dir = join(PATH.SOLVER, event_dirname(itask + 1), 'traces', 'syn')
            try:
                ev_precond = self.load(join(syn_dir, filename))
            except IOError:
                print('Could not open precond.bin for task {}'.format(itask))
            else:
                precond += ev_precond
                self.save(join(PATH.GRAD, filename), precond)

        # sum kernels
        for par in self.parameters:
            filename = par + '_kernel.bin'
            gradient = np.zeros((p.nz * p.nx), dtype='float32')
            for itask in range(PAR.NTASK):
                syn_dir = join(PATH.SOLVER, event_dirname(itask + 1), 'traces', 'syn')
                try:
                    ev_grad = self.load(join(syn_dir, filename))
                except IOError:
                    print('Could not open {} for task {}'.format(filename, itask))
                else:
                    gradient += ev_grad
                    self.save(join(PATH.GRAD, filename), gradient)


    def smooth(self, precond=False, span=0.):
        """ Process gradient
        """

        if precond:
            print('Applying preconditioner..')
            Pr = self.load(join(PATH.GRAD, 'precond.bin'))
            Pr = Pr.reshape((p.nz, p.nx))
        else:
            Pr = 1

        for par in self.parameters:
            filename = par + '_kernel.bin'
            g = self.load(join(PATH.GRAD, filename))
            g = g.reshape((p.nz, p.nx))
            g *= Pr
            gs = gridsmooth(g, span)
            self.save(join(PATH.GRAD, par + '_smooth_kernel.bin'), gs)

        g_new = self.merge(PATH.GRAD, '_smooth_kernel.bin')
        savenpy(join(PATH.OPTIMIZE, 'g_new'), g_new)

    # solver specific utilities

    def load(self, filename):
        """ Loads a float32 numpy 2D array
        """
        arr = np.fromfile(filename, dtype='float32')

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
            v = extend_pml_velocities(v, p.nx, p.nz, p.ncpml, p.use_cpml_left, p.use_cpml_right, p.use_cpml_top,
                                           p.use_cpml_bottom)

            filename = join(path, par + suffix)
            self.save(filename, v)
            ipar += 1

    # solver specific format routines

    def set_source_array(self):
        """ Generate 1D line of sources using LINE_START and DSRC.
            stores self.sources as a list of tuples (xs, zs).
        """

        if PAR.USE_SRC_FILE:
            self.sources = np.loadtxt(join(PATH.SUBMIT, 'sources'))
        else:

            line = [PAR.LINE_START + PAR.DSRC * itask for itask in range(PAR.NTASK)]
            fixed = [PAR.FIXED_POS] * PAR.NTASK

            if PAR.LINE_DIR == 'x':
                self.sources = list(zip(line, fixed))
            elif PAR.LINE_DIR == 'z':
                self.sources = list(zip(fixed, line))
            else:
                raise ValueError('Direction should be x or z')

        self.check_source_in_grid()

    def check_source_in_grid(self):
        """ Checks whether generated sources are within grid.
        """

        xmax = (p.nx * p.dx) - (int(p.use_cpml_left) + int(p.use_cpml_right)) * p.ncpml * p.dx
        zmax = (p.nz * p.dz) - (int(p.use_cpml_top) + int(p.use_cpml_bottom)) * p.ncpml * p.dz

        for itask in range(PAR.NTASK):
            xsrc = self.sources[itask][0]
            zsrc = self.sources[itask][1]
            if xsrc < 0 or xsrc > xmax:
                raise ValueError('xsrc: {} for task {}, not in grid'.format(xsrc, itask))
            if zsrc < 0 or zsrc > zmax:
                raise ValueError('zsrc: {} for task {}, not in grid'.format(zsrc, itask))

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
        write_cfg_file(join(self.getpath,'INPUT', 'par.cfg'), cfg)

    def set_src_cfg(self, **kwargs):
        """ Sets source cfg file for solver. Keys must match solver src.cfg file parameters.
        """

        # read par dict
        cfg = read_cfg_file(join(PATH.SOLVER_INPUT, 'src_template.cfg'))

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
        write_cfg_file(join(self.getpath, 'INPUT', 'src.cfg'), cfg)


    def run(self, script, output='/dev/null'):
        """ Wrapper for mpirun
        """
        with open(output,'w') as f:
            subprocess.call(
                script,
                shell=True,
                stdout=f)

    @property
    def getpath(self):
        itask = system.getnode()
        return join(PATH.SOLVER, event_dirname(itask + 1))
