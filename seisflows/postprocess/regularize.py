import sys

from seisflows.config import ParameterError, custom_import

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']

class regularize(custom_import('postprocess', 'pewf2d')):
    """ Adds regularization options to base class

        This parent class is only an abstract base class; see child classes
        TIKHONOV1 and TIKHONOV1 for usable regularization.

        Regularization hyperparameter is controlled by HYPERPAR (default = 0.0, 
        no regularization). Regularization is only valid for a rectangular grid.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(regularize, self).check()

        if 'HYPERPAR' not in PAR:
            setattr(PAR, 'HYPERPAR', 0.)

    def write_gradient(self, path):
        super(regularize, self).write_gradient(path)

        if not PAR.HYPERPAR:
            return

        # load current model and data objective gradient
        g = solver.load(path, suffix='_kernel')
        m = solver.load(PATH.MODELS + '/model_est', rescale=PAR.RESCALE)

        for key in solver.parameters:
                g[key] += PAR.HYPERPAR * self.nabla(m[key])

        self.save(path, solver.merge(g), backup='noregularize')

    def process_kernels(self, path, parameters):
        """ Processes kernels in accordance with parameter settings
        """
        solver.combine(path=path,
                       parameters=parameters)

    def sum_residuals(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def nabla(self, m):
        raise NotImplementedError("Must be implemented by subclass.")
