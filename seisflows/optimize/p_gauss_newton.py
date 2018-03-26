import sys
from glob import glob
from os.path import join

from seisflows.tools import unix
from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class p_gauss_newton(custom_import('optimize', 'p_newton')):
    """ Implements Gauss-Newton-CG algorithm.
        Computes Hessian-vector products using second-order adjoints.
        Perturbation wavefield is computed using a finite difference
        approximation of the form:
            du = (1/h) * (u(m+h*dm) - u(m))
    """

    def check(cls):
        """ Checks parameters and paths
        """
        super(p_gauss_newton, cls).check()


    def hessian_product(cls, h):
        return cls.load('g_lcg')/h


    def apply_hess(self, path=''):
        """ Computes action of Hessian on a given model vector.
        """
        system = sys.modules['seisflows_system']

        system.run_single('optimize', 'call_solver_hess',
                          model_dir=PATH.HESS+'/model')
        system.run('optimize', 'prepare_apply_hess')
        system.run_single('optimize', 'call_solver_hess',
                          model_dir=PATH.MODEL_EST,
                          adjoint=True)


    def call_solver_hess(self, model_dir, adjoint=False):
        """ Used to compute action of the Hessian on a model perturbation.
        """
        solver = sys.modules['seisflows_solver']

        if not adjoint:
            mode=0
            run_solver = solver.forward
        else:
            mode = 2
            run_solver = solver.adjoint

        solver.set_par_cfg(external_model_dir=model_dir,
                           output_dir=PATH.HESS,
                           mode=mode,
                           use_stf_file=PAR.USE_STF_FILE,
                           stf_file='stf_f.txt')
        run_solver()


    def prepare_apply_hess(self):
        """ Prepares solver to compute action of Hessian by writing adjoint traces.
            f = u(m+dm) - u(m)
        """
        solver = sys.modules['seisflows_solver']
        preprocess = sys.modules['seisflows_preprocess']

        path1 = solver.getpath
        path2 = solver.get_altpath(PATH.HESS)

        for filename in solver.data_filenames:
            obs = preprocess.reader(path1+'/'+'traces/syn', filename)
            syn = preprocess.reader(path2+'/'+'traces/syn', filename)

            obs = preprocess.process_traces(obs)
            syn = preprocess.process_traces(syn)

            preprocess.write_adjoint_traces(path2+'/'+'traces/adj', syn, obs, filename)

        # copy boundary files for reconstruction of synthetic wavefield
        src = glob(join(path1, 'traces/syn/proc*'))
        dst = join(path2, 'traces/syn')
        unix.cp(src, dst)

        src = glob(join(path1, 'traces/syn/*.su'))
        dst = join(path2, 'traces/obs')
        unix.cp(src, dst)