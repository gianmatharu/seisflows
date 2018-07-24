
import sys
import numpy as np

from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import
from seisflows.plugins import optimize

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class p_newton(custom_import('optimize', 'newton')):
    """ Implements Newton-CG algorithm.
        Computes Hessian-vector products using finite-difference
        approximation of the form:
            H(m)dm = (1/h) * (g(m+h*dm) - g(m))

        Regularization is not yet implemented with TN methods.
        Gradient smoothing is applicable. Hessians that are not
        positive definite are neglected by testing for negative
        curvature.
    """
    def check(cls):
        """ Checks parameters and paths
        """
        super(p_newton, cls).check()

        if PAR.POSTPROCESS in ['regularize', 'tikhonov0', 'tikhonov1']:
            raise NotImplementedError('Newton methods not implemented with regularization.')

    def apply_hessian(self, m, dm, h):
        """ Computes the action of the Hessian on a given vector through
          solver calls
        """
        solver = sys.modules['seisflows_solver']
        postprocess = sys.modules['seisflows_postprocess']

        self.save('m_lcg', m + h*dm)
        solver.rsave(PATH.OPTIMIZE+'/m_lcg', PATH.HESS+'/'+'model')

        self.apply_hess(path=PATH.HESS)

        postprocess.write_hessprod(path=PATH.HESS+'/'+'gradient',
                                   solver_path=PATH.HESS)

        self.save('g_lcg', solver.merge(solver.load(
                PATH.HESS+'/'+'gradient', suffix='_kernel')))

        unix.rm(PATH.HESS+'_debug')
        unix.mv(PATH.HESS, PATH.HESS+'_debug')
        unix.mkdir(PATH.HESS)

        #unix.rm(PATH.HESS)
        #unix.mkdir(PATH.HESS)

        return self.hessian_product(h)


    def hessian_product(self, h):
        solver = sys.modules['seisflows_solver']

        # load unscaled gradient
        if PAR.RESCALE:
            g = solver.merge(solver.load(PATH.GRAD, suffix='_kernel_noscale'))
        else:
            g = solver.merge(solver.load(PATH.GRAD, suffix='_kernel'))

        # load perturped gradient
        gdm = self.load('g_lcg')

        # finite difference approximation
        Hdm = solver.split((gdm - g) / h)

        if PAR.RESCALE:
            Hdm = solver.rescale.rescale_hessian_kernel(Hdm)

        return solver.merge(Hdm)


    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']

        system.run_single('optimize', 'call_solver_hess')
        system.run('optimize', 'prepare_apply_hess')
        system.run_single('optimize', 'call_solver_hess', adjoint=True)


    def call_solver_hess(self, adjoint=False):
        """ Used to compute action of the Hessian on a model perturbation.
        """
        solver = sys.modules['seisflows_solver']

        if not adjoint:
            mode=1
            run_solver = solver.forward
        else:
            mode = 2
            run_solver = solver.adjoint

        solver.set_par_cfg(external_model_dir=PATH.HESS+'/model',
                           output_dir=PATH.HESS,
                           mode=mode,
                           use_stf_file=PAR.USE_STF_FILE,
                           stf_file='stf_f.txt')
        run_solver()


    def prepare_apply_hess(self):
        """ Prepares solver to compute action of Hessian by writing adjoint traces.
            f = u(m+dm) - d
        """
        solver = sys.modules['seisflows_solver']
        preprocess = sys.modules['seisflows_preprocess']

        path1 = solver.getpath
        path2 = solver.get_altpath(PATH.HESS)

        for filename in solver.data_filenames:
            obs = preprocess.reader(path1+'/'+'traces/obs', filename)
            syn = preprocess.reader(path2+'/'+'traces/syn', filename)

            obs = preprocess.process_traces(obs, filter=not PAR.PREFILTER)
            syn = preprocess.process_traces(syn)

            preprocess.write_adjoint_traces(path2+'/'+'traces/adj', syn, obs, filename)
