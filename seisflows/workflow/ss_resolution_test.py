
import sys
from glob import glob
from os.path import join

from seisflows.config import ParameterError
from seisflows.tools.tools import exists
from seisflows.plugins.solver.pewf2d import Par
from seisflows.tools.tools import savetxt, loadtxt
from seisflows.plugins.io.pewf2d import mread, mwrite, read, write
from seisflows.plugins.misc import resolution
from seisflows.tools import unix


PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']
optimize = sys.modules['seisflows_optimize']
preprocess = sys.modules['seisflows_preprocess']
postprocess = sys.modules['seisflows_postprocess']

p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))

class ss_resolution_test(object):
    """ Resolution test base class.

      Computes Hessian vector products to approximate point-spread functions.
      Hessian vector products are approximated by a finite difference
      approximation. Can also be used to compute additional resolution proxies.
    """

    def check(self):
        """ Checks parameters and paths
        """
        # check paths
        if 'DATA' not in PATH:
            raise ParameterError(PATH, 'DATA')

        if 'GRAD' not in PATH:
            setattr(PATH, 'GRAD', join(PATH.SCRATCH, 'evalgrad'))

        if 'GRAD_FINAL' not in PATH:
            setattr(PATH, 'GRAD_FINAL', None)

        if 'HESSPROD' not in PATH:
            setattr(PATH, 'HESSPROD', join(PATH.SCRATCH, 'hessprod'))

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
        if not PAR.USE_STF_FILE:
            raise ValueError('Must use stf for gradient calculations.')
        else:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

        if PAR.SYSTEM not in ['serial', 'westgrid']:
            raise ValueError('Use system class "serial" here.')

        if PAR.SOLVER != 'ssewf2d':
            raise ValueError('Use solver class "pewf2d" here.')

        if PAR.PREPROCESS != 'ssewf2d':
            raise ValueError('Use preprocess class "ssewf2d" here.')

        # check that test type parameters are included
        if 'TEST' not in PAR:
            raise ParameterError(PAR, 'TEST_TYPE')

        if PAR.TEST not in ['Volume', 'Spike', 'Random']:
            raise ValueError('Requested test "{}" not implemented'.format(PAR.TEST))

        if 'PERC_PERT' not in PAR:
            setattr(PAR, 'PERC_PERT', None)

        if PAR.TEST == 'Spike':
            if 'SPIKE_X' not in PAR:
                raise ParameterError(PAR, 'SPIKE_X')
            else:
                resolution.check_spike_pars(PAR.SPIKE_X, p.nx)

            if 'SPIKE_Z' not in PAR:
                raise ParameterError(PAR, 'SPIKE_Z')
            else:
                resolution.check_spike_pars(PAR.SPIKE_Z, p.nz)


    def main(self):
        """ Perform resolution test
        """
        # clean directories
        self.clean_directory(PATH.OUTPUT)
        self.clean_directory(PATH.SCRATCH)

        # setup
        preprocess.setup()
        postprocess.setup()
        system.run('solver', 'setup',
                   hosts='all')

        system.run('solver', 'setup_encoding',
                   hosts='head')

        print('Generate encoding scheme...')
        system.run('solver', 'generate_encoding',
                   hosts='head')

        print('Encoding data...')
        system.run('solver', 'generate_data',
                   hosts='all')

        # make gradient and hessian-product directories
        self.clean_directory(PATH.GRAD)
        self.clean_directory(PATH.HESSPROD)

        # compute gradient in final iteration model
        print('Computing gradient in final model...')

        if PATH.GRAD_FINAL:
            self.copy_gradient()
        else:
            self.compute_gradient('grad')

        # probe Hessian for all parameters
        for par in solver.parameters:
            print('Computing gradient for perturbation in {}'.format(par))

            # refresh model
            self.setup_model(par)

            # apply perturbation
            self.perturb_model(par)

            # compute gradient
            self.compute_gradient(par)

            # compute Hessian vector products using FD approximation
            self.compute_FD_approximation(par)

        print('Finished')

    def compute_gradient(self, par):
        # generate synthetics in perturbed model

        print('Generating synthetics...')
        system.run('solver', 'generate_synthetics',
                    mode=1,
                    hosts='head')

        print('Prepare adjoint sources...')
        system.run('solver', 'prepare_eval_grad',
                   hosts='all')

        print('Computing gradient...')
        system.run('solver', 'compute_gradient',
                    hosts='head')

        # write gradient
        postprocess.combine_kernels(join(PATH.GRAD, par), solver.parameters)

    def setup_model(self, par):
        """ copy model
        """
        src = glob(join(PATH.MODEL_INIT, '*.bin'))
        dst = join(PATH.MODELS, 'model_est')
        unix.cp(src, dst)

        # create unique path
        output_path = join(PATH.GRAD, par)
        if not exists(output_path):
            unix.mkdir(output_path)

    def copy_gradient(self):

        src = glob(join(PATH.GRAD_FINAL, '*.bin'))
        dst = join(PATH.GRAD, 'grad')

        if not exists(dst):
            unix.mkdir(dst)
        unix.cp(src, dst)

    def perturb_model(self, par):
        """ perturb a single model parameter
        """
        # read working model (solver)
        model = mread(PATH.MODEL_EST, ['rho', 'vp', 'vs'])

        # switch to new parametrization
        model = solver.par_map_forward(model)

        # apply unit perturbation
        if PAR.TEST == 'Volume':
            model[par], pert_val = resolution.volume(model[par], PAR.PERC_PERT)
        elif PAR.TEST == 'Spike':
            xpos = resolution.get_spike_array(PAR.SPIKE_X)
            zpos = resolution.get_spike_array(PAR.SPIKE_Z)
            model[par], pert_val = resolution.spike(model[par], (p.nx, p.nz), xpos, zpos, PAR.PERC_PERT)
        elif PAR.TEST == 'Random':
            model[par], pert = resolution.random(model[par], PAR.PERC_PERT)
            write(pert, join(PATH.GRAD, par), par, suffix='_pert')

        # write perturbed model
        mwrite(model, PATH.MODEL_EST)

        # save perturbation value to scale Hdm
        if PAR.TEST in ['Volume', 'Spike']:
            # apply parameter rescaling
            if PAR.RESCALE:
                print '{} pert_val: {}'.format(par, pert_val)
                print '{} scale_val: {}'.format(par, solver.scale[par])

                pert_val /= solver.scale[par]

            savetxt(join(PATH.GRAD, '{}_pert_val'.format(par)), pert_val)

        # revert to solver parameters
        model = solver.par_map_inverse(model)
        mwrite(model, PATH.MODEL_EST)

    def compute_FD_approximation(self, par):
        """ compute Hdm ~ g(m+dm) - g(m)
        """
        # gradient paths
        gpath = join(PATH.GRAD, 'grad')
        ppath = join(PATH.GRAD, par)

        # read gradients
        grad = mread(gpath, solver.parameters, suffix='_kernel')
        pert_grad = mread(ppath, solver.parameters, suffix='_kernel')

        # read perturbation value to scale Hdm
        if PAR.TEST in ['Volume', 'Spike']:
            pert_val = loadtxt(join(PATH.GRAD, '{}_pert_val'.format(par)))
            print('Scaling Hdm by perturbation value: {}'.format(pert_val))

        for key in solver.parameters:
            pert_grad[key] = (pert_grad[key] - grad[key])

            if PAR.RESCALE:
                pert_grad[key] *= solver.scale[key]

            if PAR.TEST in ['Volume', 'Spike']:
                pert_grad[key] /= pert_val

            write(pert_grad[key], PATH.HESSPROD, par, prefix='H_{}_'.format(key))

    def clean_directory(self, path):
        """ If dir exists clean otherwise make
        """
        if not exists(path):
            unix.mkdir(path)
        else:
            unix.rm(path)
            unix.mkdir(path)


