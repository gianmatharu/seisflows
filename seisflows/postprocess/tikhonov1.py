import sys
import numpy as np
from os.path import join

from seisflows.config import ParameterError, custom_import
from seisflows.plugins.solver.pewf2d import Par
from seisflows.tools.math import grad, nabla2

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']

system = sys.modules['seisflows_system']
solver = sys.modules['seisflows_solver']

p = Par()
p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))


class tikhonov1(custom_import('postprocess', 'regularize')):
    """ Adds regularization options to base class
 
        Available options include 0-, 1-, and 2- order Tikhonov and total
        variation regularization. While the underlying theory is classical,
        application to unstructured numerical grids via the
        "seisflows.tools.math.nabla" operator is somewhat complicated. 
 
        So far, can only be used for 2D inversion, because the required spatial
        derivative operator "nabla" is not yet available for 3D grids.
    """

    def check(self):
        """ Checks parameters and paths
        """
        super(tikhonov1, self).check()

        # PAD_LAP imposes a mask onto the laplacian operator. It
        # effectively ignores derivatives on the boundary of the model.
        # This is a means to neglect large derivatives that generate
        # instabilities near the source/receiver and boundary regions.
        if 'PAD_LAP' not in PAR:
            setattr(PAR, 'PAD_LAP', 4)

    def sum_residuals(self):
        """ Evaluate regularization term. 
           1/2 (|| DxM || + ||DzM||)
        """
        residuals = 0.
        m = solver.load(PATH.MODELS + '/model_est', rescale=PAR.RESCALE)

        for key in solver.parameters:
            m[key] = m[key].reshape((p.nz, p.nx))
            Dx = np.zeros((p.nz, p.nx))
            Dz = np.zeros((p.nz, p.nx))

            # Compute spatial derivatives
            Dx[PAR.PAD_LAP:p.nz-PAR.PAD_LAP, PAR.PAD_LAP:p.nx-PAR.PAD_LAP], \
            Dz[PAR.PAD_LAP:p.nz-PAR.PAD_LAP, PAR.PAD_LAP:p.nx-PAR.PAD_LAP] = \
                grad(m[key][PAR.PAD_LAP:p.nz-PAR.PAD_LAP,
                                 PAR.PAD_LAP:p.nx-PAR.PAD_LAP],
                                 h=[p.dx, p.dz])

            # add contribution to misfit
            residuals += 0.5 * PAR.HYPERPAR * np.sum(Dx*Dx)
            residuals += 0.5 * PAR.HYPERPAR * np.sum(Dz*Dz)

        return residuals

    def nabla(self, m):
        """ Evaluate regularization gradient term.
        """
        m = m.reshape((p.nz, p.nx))
        gm = np.zeros((p.nz, p.nx))
        gm[PAR.PAD_LAP:p.nz-PAR.PAD_LAP,
           PAR.PAD_LAP:p.nx-PAR.PAD_LAP] = nabla2(m[PAR.PAD_LAP:p.nz-PAR.PAD_LAP,
                                                    PAR.PAD_LAP:p.nx-PAR.PAD_LAP], h=[p.dx, p.dz])
        return gm.reshape((p.nz*p.nx))