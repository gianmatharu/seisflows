import sys

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']

class regularize(custom_import('postprocess', 'pewf2d')):
    """ Adds regularization options to base class

        This parent class is only an abstract base class; see child classes
        "tikhonov0" and "tikhonov1" for usable regularization.

        Regularization hyperparameter is controlled by HYPERPAR (default = 0.0,
        no regularization). Regularization is only valid for a rectangular grid.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(regularize, self).check()

        # regularization hyperparameter
        if 'HYPERPAR' not in PAR:
            setattr(PAR, 'HYPERPAR', 0.)

    def write_gradient(self, path, solver_path=''):
        """ Write gradient with contribution from regularization
        """
        super(regularize, self).write_gradient(path, solver_path=solver_path)

        if not PAR.HYPERPAR:
            return

        # load current model and data objective gradient
        g = solver.load(path, suffix='_kernel')
        m = solver.rload(PATH.MODEL_EST)

        if PAR.DEBUG:
            print 'Max of gradient: {:6e}'.format(abs(solver.merge(g)).max())

        reg = {}
        for key in solver.parameters:
                if PAR.DEBUG:
                    print 'Max of {} gradient: {:.6e}'.format(key, abs(self.nabla(m[key], key)).max())
                reg[key] = self.nabla(m[key], key)
                g[key] += PAR.HYPERPAR * self.nabla(m[key], key)

        # backup kernels
        self.save(path, solver.merge(g), backup='noregularize')

        # save contribution from regularization
        if PAR.DEBUG:
            solver.save(reg, path, suffix='_reg')

    def process_kernels(self, path, parameters, solver_path=''):
        """ Processes kernels in accordance with parameter settings
        """
        solver.combine(path=path,
                       parameters=parameters,
                       solver_path=solver_path)

    def sum_residuals(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def nabla(self, m, key):
        raise NotImplementedError("Must be implemented by subclass.")