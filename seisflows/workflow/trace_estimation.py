
import sys
from glob import glob
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from seisflows.tools import unix
from seisflows.config import ParameterError
from seisflows.tools.tools import exists
from seisflows.plugins.solver.pewf2d import Par
from seisflows.tools.array import loadnpy, savenpy
from seisflows.plugins.solver_io.pewf2d import mread, mwrite, write
from seisflows.workflow.base import base


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']

p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class trace_estimation(base):
    """ Stochastic trace estimator.

      Computes stochastic approximations to the diagonals of the Hessian.
      Uses Newton/Gauss-Newton approximations. Estimates the traces of
      sub-Hessians for each source also.
    """

    def check(self):
        """ Checks parameters and paths
        """
        # check paths
        if 'DATA' not in PATH:
            raise ParameterError(PATH, 'DATA')

        if 'GRAD' not in PATH:
            setattr(PATH, 'GRAD', join(PATH.SCRATCH, 'evalgrad'))

        if 'HESS' not in PATH:
            setattr(PATH, 'HESS', join(PATH.SCRATCH, 'evalhess'))

        if 'OPTIMIZE' not in PATH:
            setattr(PATH, 'OPTIMIZE', join(PATH.SCRATCH, 'optimize'))

        if 'MODEL_INIT' not in PATH:
            raise ParameterError(PATH, 'MODEL_INIT')

        if 'MODEL_TRUE' not in PATH:
            raise ParameterError(PATH, 'MODEL_TRUE')

        if 'MODELS' not in PATH:
            setattr(PATH, 'MODELS', join(PATH.SUBMIT, 'models'))

        if 'MODEL_EST' not in PATH:
            setattr(PATH, 'MODEL_EST', join(PATH.MODELS, 'model_est'))

        # check parameters
        if 'NTRIALS' not in PAR:
            setattr(PAR, 'NTRIALS', 5)

        if 'STORE_IND' not in PAR:
            setattr(PAR, 'STORE_IND', False)

        if 'DEBUG' not in PAR:
            setattr(PAR, 'DEBUG', False)

        if not PAR.USE_STF_FILE:
            raise ValueError('Must use stf for gradient calculations.')
        else:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

        if PAR.SYSTEM not in ['serial']:
            raise ValueError('Use system class "serial" here.')

        if PAR.SOLVER != 'pewf2d':
            raise ValueError('Use solver class "pewf2d" here.')

        if PAR.OPTIMIZE not in ['p_gauss_newton', 'p_newton']:
            raise ValueError('Must use a Newton based class here')

    def main(self):
        """ Perform stochastic trace estimation
        """
        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)
        self.clean_directory(PATH.GRAD)
        self.clean_directory(PATH.HESS)

        if PAR.DEBUG:
            self.test_estimation()
            return

        # setup
        system.run('solver', 'setup')
        preprocess.setup()
        postprocess.setup()
        optimize.setup()

        # compute synthetic wavefield
        print('Generating synthetics...')
        system.run_single('solver', 'generate_synthetics', mode=1)

        # initialize trace and diagonal
        diag, trace, traces = {}, {}, {}
        for par in solver.parameters:
            trace[par] = 0.
            traces[par] = np.zeros(PAR.NTASK)
            diag[par] = np.zeros((p.nx*p.nz))

        # perform repeated trials for stochastic trace estimation
        for i in range(PAR.NTRIALS):
            print("Running trial {} of {}".format(i+1, PAR.NTRIALS))

            for par in solver.parameters:
                print('Running trials for {}'.format(par))
                base_dir = PATH.SCRATCH+'/'+'{:03d}/{}'.format(i+1, par)
                self.clean_directory(base_dir)

                # generate random vector
                m, dm = self.perturb_model(base_dir, par)

                # apply Hessian
                print("Computing Hessian-vector product")
                Hdm = self.apply_hessian(m, dm, base_dir)

                # export Hessian-vector products
                self.export_products(Hdm, base_dir, store_individual=PAR.STORE_IND)

                # load as dictionaries
                dm = solver.split(self.load(base_dir, 'dm'))
                Hdm = solver.split(self.load(base_dir, 'Hdm'))

                # estimate trace and diagonal
                diag[par] += (Hdm[par] * dm[par]) / (dm[par] * dm[par])
                trace[par] += (dm[par].dot(Hdm[par])) / (dm[par].dot(dm[par]))
                local_trace = (p.nx*p.nz) * ((dm[par].dot(Hdm[par])) / (dm[par].dot(dm[par])))

                # scaling
                current_trace = ((p.nx*p.nz) / (i+1)) * trace[par]
                print('[ITER {:03d}] - {} local estimate: {:.6e}'.format(i+1, par, local_trace))
                print('[ITER {:03d}] - {} trace estimate: {:.6e}'.format(i+1, par, current_trace))

        # Finalize diagonal + trace estimates
        unix.mkdir(PATH.SCRATCH+'/'+'trace')
        for par in solver.parameters:
            diag[par] /= PAR.NTRIALS
            write(diag[par], PATH.SCRATCH+'/'+'trace', par)

            trace[par] *= ((p.nx*p.nz) / (PAR.NTRIALS))
            print('Final {} trace estimate: {:.6e}'.format(par, trace[par]))

        # # compute Prob dist
        for itask in range(PAR.NTASK):
            for i in range(PAR.NTRIALS):
                for par in solver.parameters:
                    base_dir = PATH.SCRATCH+'/'+'{:03d}/{}'.format(i+1, par)
                    dm = solver.split(self.load(base_dir, 'dm'))
                    Hdm = solver.split(self.load(base_dir, 'Hdm_{:03d}'.format(itask+1)))

                    traces[par][itask] += (dm[par].dot(Hdm[par])) / (dm[par].dot(dm[par]))

            for par in solver.parameters:
                traces[par][itask] *= ((p.nx*p.nz) / (PAR.NTRIALS))

        total_trace = 0
        local_trace = np.zeros(PAR.NTASK)
        for par in solver.parameters:
            local_trace += traces[par]
            total_trace += np.sum(traces[par])

        prob = local_trace / total_trace
        print prob, np.sum(prob)
        self.save(PATH.SCRATCH +'/trace', 'PROB_DIST', prob)
        plt.bar(range(PAR.NTASK), prob)
        plt.show()
        print('Finished')

    def test_estimation(self):
        """ Test trace estimation algorithm.
        """
        n = 100
        H = np.diag(np.random.rand(n))

        trace = 0
        for i in range(PAR.NTRIALS):
            dm = np.random.choice([-1, 1], n)
            trace += (dm.dot(H.dot(dm)) / dm.dot(dm))

        print('Estimated trace of H: {:.4e}'.format((n/PAR.NTRIALS)*trace))
        print('Actual trace of H: {:.4e}'.format(np.trace(H)))
        return

    def perturb_model(self, path, par):
        """ perturb model with a random vector
        """
        # read working model (solver)
        model = mread(PATH.MODEL_EST, ['rho', 'vp', 'vs'])

        # switch to new parametrization
        model = solver.par_map_forward(model)

        # store random vector
        dm = {}
        # apply Rademacher random vector
        for key in model:
            if key == par:
                dm[key] = np.random.choice([-1, 1], p.nx*p.nz)
            else:
                dm[key] = np.zeros((p.nx*p.nz))

        # save random vector
        self.save(path, 'dm', solver.merge(dm))

        return solver.merge(model), solver.merge(dm)

    def export_products(self, Hdm, path, store_individual=False):
        """ Export Hessian-vector products
        """
        self.save(path, 'Hdm', Hdm)

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """
        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)

    def load(self, path, filename):
        # reads vectors from disk
        return loadnpy(path+'/'+filename)

    def save(self, path, filename, array):
        # writes vectors to disk
        savenpy(path+'/'+filename, array)

    def apply_hessian(self, m, dm, path):
        """ Trace estimation with the Gauss-Newton Hessian
        """
        # Save peturbed model
        h = PAR.EPSILON * max(abs(m))/max(abs(dm))
        optimize.save('m_lcg', m + h*dm)
        solver.rsave(PATH.OPTIMIZE+'/m_lcg', PATH.HESS+'/'+'model')

        # apply Gauss-Newton Hessian
        system.run_single('optimize', 'call_solver_hess',
                          model_dir=PATH.HESS+'/model')
        system.run('optimize', 'prepare_apply_hess')
        system.run_single('optimize', 'call_solver_hess',
                          model_dir=PATH.MODEL_EST,
                          adjoint=True)

        postprocess.write_gradient(path=PATH.HESS+'/'+'gradient',
                                   solver_path=PATH.HESS)

        # Compute Hessian-vector product
        Hdm = solver.merge(solver.load(PATH.HESS+'/'+'gradient', suffix='_kernel'))/h

        # compute per source products
        for i in range(PAR.NTASK):
            src = PATH.HESS + '/{:03d}/traces/syn'.format(i+1)
            Hdm_l = solver.merge(solver.load(src, suffix='_kernel')) / h
            self.save(path, 'Hdm_{:03d}'.format(i+1), Hdm_l)

        # cleanup
        unix.rm(PATH.HESS)
        unix.mkdir(PATH.HESS)

        return Hdm