from os.path import join, basename

import numpy as np
from glob import glob
import subprocess

from seisflows.seistools.ewf2d import Par, read_cfg_file, write_cfg_file, event_dirname
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

        if 'PRECOND_TYPE' not in PAR:
            setattr(PAR, 'PRECOND_TYPE', 'LOCAL')

        if 'PRECOND_THRESH' not in PAR:
            setattr(PAR, 'PRECOND_THRESH', 1e-4)

        if 'PRECOND_SMOOTH' not in PAR:
            setattr(PAR, 'PRECOND_SMOOTH', None)

        if 'SAFEUPDATE' not in PAR:
            setattr(PAR, 'SAFEUPDATE', None)

        if PAR.SAFEUPDATE:
            if 'VPMIN' not in PAR:
                raise ParameterError(PAR, 'VPMIN')

            if 'VPMAX' not in PAR:
                raise ParameterError(PAR, 'VPMAX')

            if PAR.MATERIALS == 'Elastic':
                if 'VSMIN' not in PAR:
                    raise ParameterError(PAR, 'VSMIN')

                if 'VSMAX' not in PAR:
                    raise ParameterError(PAR, 'VSMAX')

        self.set_source_array()

    ### low level interface

    def forward(self):
        """ Perform forward simulation. Must launch from /bin.
        """
        unix.cd(join(self.getpath, 'bin'))
        script = './xewf2d'
        super(ewf2d, self).call(script, PATH.SUBMIT + '/dump')

        unix.cd(PATH.SUBMIT)

    def adjoint(self):
        """ Perform adjoint simulation. Must launch from /bin
        """
        unix.cd(join(self.getpath, 'bin'))
        script = './xewf2d'
        super(ewf2d, self).call(script, PATH.SUBMIT + '/dump')

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
            self.generate_data(model_dir=PATH.MODEL_TRUE, output_dir=output_dir, use_stf_file=PAR.USE_STF_FILE,
                               stf_file=PAR.STF_FILE)

    def generate_synthetics(self):
        # generate synthetic data
        output_dir = join(self.getpath, 'traces', 'syn')
        self.generate_data(model_dir=PATH.MODEL_INIT, output_dir=output_dir, save_wavefield=False,
                           use_stf_file=PAR.USE_STF_FILE, stf_file=PAR.STF_FILE)

    def compute_gradient(self):
        """ Sequential event gradient computation, reduces storage requirement.
        """

        # generate synthetic data
        output_dir = join(self.getpath, 'traces', 'syn')
        self.generate_data(model_dir=PATH.MODEL_EST, output_dir=output_dir, save_wavefield=True,
                           use_stf_file=PAR.USE_STF_FILE, stf_file='stf_f.txt')

        # prepare adjoint sources
        preprocess.prepare_eval_grad(self.getpath)

        # compute event gradient
        self.evaluate_gradient(model_dir=PATH.MODEL_EST)

    def evaluate_function(self, path=''):
        """ Evaluate test function
        """
        # generate synthetic data
        itask = system.getnode()

        basedir = join(path, event_dirname(itask + 1))
        output_dir = join(basedir, 'traces', 'syn')
        model_dir = join(path, 'model')
        unix.mkdir(output_dir)

        self.generate_data(model_dir=model_dir, output_dir=output_dir, save_wavefield=True,
                           use_stf_file=PAR.USE_STF_FILE, stf_file='stf_f.txt')

        preprocess.evaluate_trial_step(self.getpath, basedir)

    # frugal inversion functions

    def fg_compute_gradient(self):
        """ Evaluate gradient in frugal inversion.
        """
        # prepare adjoint sources
        preprocess.prepare_eval_grad(self.getpath)

        # compute event gradient
        self.evaluate_gradient(model_dir=PATH.MODEL_EST)

    def export_trial_solution(self, path=''):
        """ Save trial solution for frugal inversion.
        """
        # transfer synthetic data
        src = glob(join(path, basename(self.getpath), 'traces', 'syn', '*'))
        dst = join(self.getpath, 'traces', 'syn')
        unix.mv(src, dst)

    # per event functions

    def generate_data(self, model_dir=PATH.MODEL_TRUE, output_dir='', save_wavefield=False, use_stf_file=False,
                      stf_file=''):
        """ Generate dataset. Defaults to generating synthetic data for true model.
        """

        # get task number
        itask = system.getnode()

        # set par.cfg file for solver
        self.set_par_cfg(external_model_dir=model_dir, output_dir=output_dir, save_final_wavefield=save_wavefield)

        # set src.cfg for solver
        xsrc = self.sources[itask][0]
        zsrc = self.sources[itask][1]
        self.set_src_cfg(xs=float(xsrc), zs=float(zsrc), use_stf_file=use_stf_file, stf_file=stf_file)

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

    # serial/reduction function

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

        xmax = (p.nx * p.dx)
        zmax = (p.nz * p.dz)

        for itask in range(PAR.NTASK):
            xsrc = self.sources[itask][0]
            zsrc = self.sources[itask][1]
            if xsrc < 0 or xsrc > xmax:
                raise ValueError('xsrc: {} for task {}, not in grid'.format(xsrc, itask))
            if zsrc < 0 or zsrc > zmax:
                raise ValueError('zsrc: {} for task {}, not in grid'.format(zsrc, itask))

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

    def apply_preconditioner(self, dir, grad):
        """ Prepare preconditioner
        """

        if PAR.PRECOND_TYPE == 'LINEAR':
            grad = grad.reshape((p.nz, p.nx))
            precond = np.linspace(0, 1, p.nz)
            grad = (grad.T * precond).T
            grad = grad.reshape(p.nz * p.nx)
            return grad
        if PAR.PRECOND_TYPE == 'LOCAL':
            precond = self.load(join(dir, 'precondf.bin'))
            return precond * grad
        elif PAR.PRECOND_TYPE == 'ONE_WAY':
            precond = self.load(join(dir, 'precond.bin'))
            precond = self._invert_grid_interior(precond)
            return precond * grad
        elif PAR.PRECOND_TYPE == 'TWO_WAY':
            precond = abs(self.load(join(dir, 'precond2w.bin')))
            precond = self._invert_grid_interior(precond, smooth=True)
            return precond * grad
        else:
            raise ValueError('Preconditioner type not found.')

    def _invert_grid_interior(self, x, smooth=False):
        """ Invert grid interior
        """
        x = x.reshape((p.nz, p.nx))

        if smooth:
            x = gridsmooth(x, PAR.PRECOND_SMOOTH)

        # normalize initial input
        x /= abs(x).max()
        x = 1 / (x + PAR.PRECOND_THRESH)
        x /= abs(x).max()
        x = x.reshape(p.nz * p.nx)

        return x


    @property
    def getpath(self):
        itask = system.getnode()
        return join(PATH.SOLVER, event_dirname(itask + 1))
