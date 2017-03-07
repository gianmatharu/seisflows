import numpy as np
import skimage.feature as ski
from scipy import ndimage as ndi
from scipy.ndimage import correlate1d, gaussian_filter1d
from scipy.sparse.linalg import LinearOperator, cg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from seisflows.tools.math import eigsorted
from scipy.ndimage.filters import laplace
from seisflows.tools.array import check_2d, as_ndarrays
from functools import partial
import sys


# class definitions

class StructureTensor2D(object):
    """ 2D structure tensor class.
    """

    def __init__(self, t11, t12, t22):

        self.check(t11, t12, t22)

        # Initialize structure tensor
        self.t11 = t11
        self.t12 = t12
        self.t22 = t22

    def check(self, t11, t12, t22):
        # verify input
        t11, t12, t22 = as_ndarrays(t11, t12, t22)
        check_2d(t11, t12, t22)

    # alternate constructor methods.
    @classmethod
    def fromimage(cls, image, sigma=1.0, mode='constant', cval=0.0, coherent=False):
        """ Construct a structure tensor from an input image.

        Parameters
        ----------
        image: array_like
            Input image.
        sigma: float, optional.
                standard deviation of Gaussian kernel.
        mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.
        cval: float, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.
        coherent: bool, optional
            If set to false, regular structure tensors are computed. If set to true, a
            diffusion tensor is returned. Diffusion tensors share eigenvectors with
            a structure tensor but have eigenvalues largest in directions of least change.
            Useful for structure oriented smoothing. Note diffusion tensors are singular, for
            more diffusion tensors set coherent=False and then call invert_structure(p0, p1).

        Returns
        -------
        output: StructureTensor2D
            Instance of structure tensor.
        """

        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError('Image must be 2-dimensional!')

        if not coherent:
            image = image / abs(image.max())
            t11, t12, t22 = ski.structure_tensor(image, sigma=sigma, mode=mode, cval=cval)
        else:
            t11, t12, t22 = cls.diffusion_tensor(image, sigma=sigma, mode=mode, cval=cval)

        return cls(t11, t12, t22)

    @classmethod
    def isotropic(cls, nx, ny):
        """  Sets an isotropic, spatially-invariant structure tensor field.

        Parameters
        ----------
        nx, ny: int, required.
            dimensions of 2D field.

        Returns
        -------
        output: StructureTensor2D
            Instance of structure tensor.
        """
        return cls(np.ones((ny, nx)), np.zeros((ny, nx)), np.ones((ny, nx)))

    def get_tensor(self):

        return self.t11, self.t12, self.t22

    def get_tensor_at(self, ix, iy):

        return self.t11[iy, ix], self.t12[iy, ix], self.t22[iy, ix]

    def invert_structure(self, p0, p1):
        """ From Mine JTK toolkit. Inverts structure tensor by inverting
        eigenvalues.

        Inverts these tensors, assumed to be structure tensors. After inversion,
        all eigenvalues are in the range (0,1]. Specifically, after inversion,
        0 < au <= av <= 1.

        Before inversion, tensors are assumed to be structure tensors,
        for which eigenvalues au are not less than their corresponding eigenvalues av.
        (Any eigenvalues au for which this condition is not satisfied are set equal to
        the corresponding eigenvalue av.)

        Then, if any eigenvalues are equal to zero, this method adds a small fraction
        of the largest eigenvalue au to all eigenvalues. If am is the minimum of the
        eigenvalues av after this perturbation, then the parameter p0 is used to
        compute a0 = pow(am/av,p0) and the parameter p1 is used to compute
        a1 = pow(av/au,p1). Inverted eigenvalues are then au = a0*a1 and av = a0.

        In this way, p0 emphasizes overall amplitude and p1 emphasizes linearity.
        For amplitude-independent tensors with all eigenvalues av equal to one,
        set p0 = 0.0. To enhance linearity, set p1 > 1.0. To simply invert (and normalize)
        these tensors, set p0 = p1 = 1.0.
        """

        au, av, u1, u2 = self.get_eigvecvals()

        amax = 0.0
        amin = sys.float_info.max

        ny, nx = au.shape

        # Adjust eigenvalues
        for i in range(ny):
            for j in range(nx):
                aul = au[i, j]
                avl = av[i, j]

                if avl < 0:
                    avl = 0.0
                if aul < avl:
                    aul = avl
                if avl < amin:
                    amin = avl
                if aul > amax:
                    amax = aul

                au[i, j] = aul
                av[i, j] = avl

        aeps = max(sys.float_info.min * 100, sys.float_info.epsilon*amax)
        amin += aeps
        amax += aeps

        for i in range(ny):
            for j in range(nx):
                aul = au[i, j] + aeps
                avl = av[i, j] + aeps
                a0l = pow(amin / avl, p0)
                a1l = pow(avl / aul, p1)

                au[i, j] = a0l * a1l
                av[i, j] = a0l

        self.t11, self.t12, self.t22 = self.tensor_from_eig(au, av, u1, u2)

    @staticmethod
    def diffusion_tensor(image, sigma=1.0, mode='constant', cval=0.0, p0=1.0, p1=1.0):
        """ Compute spatially variant diffusion tensor.

        For a given image I, the structure tensor is given by

        | Ixx Ixy |
        | Ixy Iyy |

        with eigenvectors u and v. Eigenvectors v correspond to the smaller
        eigenvalue and point in directions of maximum coherence.
        The diffusion tensor D is given by, D = v * transpose(v).
        This tensor is designed for anisotropic smoothing.
        Local D tensors are singular.

        Parameters
        ----------
        image: array_like
            Input image.
        sigma: float, optional
            Standard deviation of Gaussian kernel.
        mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.
        cval:
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.

        Returns
        -------
        d11, d12, d22: ndarray
            Independent components of diffusion tensor.
        """

        image = np.asarray(image)

        if image.ndim != 2:
            raise ValueError('Image must be 2-dimensional!')

        # normalize image
        image = image / abs(image.max())
        (ny, nx) = image.shape

        # # Initialize diffusion tensor components
        d11 = np.zeros((ny, nx))
        d12 = np.zeros((ny, nx))
        d22 = np.zeros((ny, nx))

        # Compute gradient of scalar field using derivative filter
        Ixx, Ixy, Iyy = ski.structure_tensor(image, sigma=sigma, mode=mode, cval=cval)

        for i in range(ny):
            for j in range(nx):
                A = [[Ixx[i, j], Ixy[i, j]], [Ixy[i, j], Iyy[i, j]]]
                _, vecs = eigsorted(A)

                # eigen-decomposition of diffusion tensor
                D = np.outer(vecs[:, -1], vecs[:, -1])

                # Assign components
                d11[i][j] = D[0][0]
                d12[i][j] = D[0][1]
                d22[i][j] = D[1][1]

        return d11, d12, d22

    def get_eigvecvals(self):
        """ Return local eigenvalues & largest eigenvector components of 2x2 structure tensor.
        """

        ny, nx = self.t11.shape
        au = np.zeros((ny, nx))
        av = np.zeros((ny, nx))
        u1 = np.zeros((ny, nx))
        u2 = np.zeros((ny, nx))

        for i in range(ny):
            for j in range(nx):
                A = [[self.t11[i, j], self.t12[i, j]], [self.t12[i, j], self.t22[i, j]]]
                vals, vecs = eigsorted(A)

                au[i, j] = vals[0]
                av[i, j] = vals[1]
                u1[i, j] = vecs[0, 0]
                u2[i, j] = vecs[1, 0]

        return au, av, u1, u2

    def get_determinant(self):
        """ Return local determinants of 2x2 structure tensor.
        """

        return self.t11 * self. t22 - self.t12 * self.t12

    @staticmethod
    def tensor_from_eig(au, av, u1, u2):
        """ Return local tensor components from eigenvalues and eigenvectors
        using the decomposition, T = au * uu' + vv', where eigenvalues au > av.
        A more convenient form is T = (au - av) * uu' + av * I (using identity uu' + vv' = I).

        Parameters
        ----------
        au: array_like,
            Largest eigenvalues
        av: array_like
            Smaller eigenvalues
        u1: array_like
            First component of eigenvector corresponding to largest eigenvalue.
        u2: array_like
            Second component of eigenvector corresponding to largest eigenvalue.

        Returns
        -------
        t11, t12, t22: ndarray, ndim=2
            Tensor components
        """

        # verify input
        au, av, u1, u2 = as_ndarrays(au, av, u1, u2)
        check_2d(au, av, u1, u2)

        # decomposition
        au -= av
        t11 = au * u1 * u1 + av
        t12 = au * u1 * u2
        t22 = au * u2 * u2 + av

        return t11, t12, t22


class SmoothCovariance(object):
    """ Smoothing covariance class. Designed to approximate the Matern covariance
        for 2D image processing. Allows for spatially variant, anisotropic model
        covariances.
    """

    def __init__(self, a=1.0, b=0.0, shape=1.0, sigma=1.0, range=40.0, ndim=2):

        self._ndim = ndim
        self._shape = shape
        self._sigma = sigma
        self._range = range
        self._order = int(1 + self._shape)
        self.check()

        # scaled versions of Smoothing covariance parameters.
        self._ascl = None
        self._bscl = None
        self._cscl = None
        self._kscl = None

        # Initialize approximate Matern covariance
        self._init2(a, b)

    def check(self):
        """ Check parameters of smoothing covariance.
        """
        if self._ndim != 2:
            raise NotImplementedError('Smoothing covariance only implemented for 2D')

        if self._shape <= 0 or self._shape > 3:
            raise ValueError('Shape parameter must be between 0 and 3')

        if self._range < 0:
            raise ValueError('Effective range must be > 0.')

    def _init2(self, a, b):
        """ Initialize paramters for 2D smoothing covariance.
        a: float
            Matern matching parameter
        b: float
            Matern matching parameter
        """
        self._ascl = a
        self._bscl = b
        self._cscl = (4 * np.pi * self._shape) * self._sigma * self._sigma
        self._kscl = 0.5 * self._range / np.sqrt(self._shape)

    def scale_parameters(self):

        akk = self._ascl * self._kscl * self._kscl
        bkk = self._bscl * self._kscl * self._kscl
        ckk = self._cscl * self._kscl * self._kscl

        return akk, bkk, ckk

    # accessor functions

    def get_shape(self):
        return self._shape

    def get_order(self):
        return self._order

    def get_sigma(self):
        return self._sigma

    def get_range(self):
        return self._range

    # main functions

    def apply(self, image, d11, d12, d22):
        """ Applies smoothing covariance to an input image.

        Parameters
        ----------
        image: array_like
            Input image.
        d11, d12, d22: array_like
            Diffusion tensor components.

        Returns
        -------
        output: ndarray, ndim=2
            Covariance applied to input image.
        """

        # verify input
        image, d11, d12, d22 = as_ndarrays(image, d11, d12,d22)
        check_2d(image, d11, d12, d22)

        # get scaled parameters
        akk, bkk, ckk = self.scale_parameters()

        # Sequentially solve second order PDE's (Hale 2014)
        #output = self._cscale(image, d11, d12, d22, ckk)
        output = image
        for i in range(self._order):
            output = _solve_directional_laplacian(output, d11, d12, d22, akk)

        if bkk != 0.0:
            output = _solve_directional_laplacian(output, d11, d12, d22, bkk)

        #output = self._cscale(output, d11, d12, d22, ckk)

        return output

    def apply_inverse(self, image, d11, d12, d22):
        """ Apply inverse of smoothing covariance to an input image.

        Parameters
        ----------
        image: array_like
            Input image.
         d11, d12, d22: array_like
            Diffusion tensor components.

        Returns
        -------
        output: ndarray, ndim=2
            Inverse covariance applied to input image.
        """

        # verify input
        image, d11, d12, d22 = as_ndarrays(image, d11, d12, d22)
        check_2d(image, d11, d12, d22)

        # get scaled parameters
        akk, bkk, ckk = self.scale_parameters()

        # Sequentially apply differential operators (Hale 2014)
        #output = self._cscale_inverse(image, d11, d12, d22, ckk)
        output = image
        if bkk != 0.0:
            output = _imapply_directional_laplacian(output, d11, d12, d22, bkk)

        for i in range(self._order):
            output = _imapply_directional_laplacian(output, d11, d12, d22, akk)

        #output = self._cscale_inverse(output, d11, d12, d22, ckk)

        return output

    def apply_half(self, image, d11, d12, d22):
        """ Applies half of the model covariance. I.e. if C = F * transpose(F),
            apply_half applies F to an input image. Only applicable for shape=1.

        Parameters
        ----------
        image: array_like
            Input image.
         d11, d12, d22: array_like
            Diffusion tensor components.

        Returns
        -------
        output: ndarray, ndim=2
            Returns model with covariance dictated by diffusion tensor.
        """

        # verify input
        if self._shape != 1:
            raise NotImplementedError('apply_half is only usable for shape = 1')

        image, d11, d12, d22 = as_ndarrays(image, d11, d12, d22)
        check_2d(image, d11, d12, d22)

        # get scaled parameters
        akk, bkk, ckk = self.scale_parameters()


        #output=image
        output = self._cscale(image, d11, d12, d22, ckk)
        output = _solve_directional_laplacian(output, d11, d12, d22, akk)
        return output


    # private functions
    def _cscale(self, image, d11, d12, d22, ckk):
        """ Apply scaling for smoothing covariance.

        Parameters
        ----------
        image: ndarray, ndim=2
           Input image.
        d11, d12, d22: ndarray, ndim=2
            Components of diffusion tensor.

        Returns
        -------
        output: ndarray, ndim=2
            Scaled image.
        """

        rckk = np.sqrt(ckk)
        det = d11 * d22 - d12 * d12
        output = rckk * np.sqrt(np.sqrt(det)) * image

        return output


    def _cscale_inverse(self, image, d11, d12, d22, ckk):
        """ Apply inverse scaling for smoothing covariance.

        Parameters
        ----------
        image: ndarray, ndim=2
           Input image.
        d11, d12, d22: ndarray, ndim=2
            Components of diffusion tensor.

        Returns
        -------
        output: ndarray, ndim=2
            Scaled image.
        """

        rckk = 1.0 / np.sqrt(ckk)
        det = 1.0 / (d11 * d22 - d12 * d12)
        output = rckk * np.sqrt(np.sqrt(det)) * image

        return output

# utility functions

def compute_derivatives(image, mode='reflect', cval=0.0):
    """ Compute gradient of 2D image with centered finite difference approximation.

    Parameters
    ----------
    image: array_like
        Input image.
    mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval: float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    imx, imy: ndarray, ndim=2
        Gradients in x, y directions respectively.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional!')
    #
    imx = correlate1d(image, [-0.5, 0, 0.5], 1, mode=mode, cval=cval)
    imy = correlate1d(image, [-0.5, 0, 0.5], 0, mode=mode, cval=cval)

    return imx, imy


def compute_hessian(image, mode='reflect', cval=0.0):
    """ Compute Hessian components of 2D image using finite difference.

    Parameters
    ----------
    image: array_like
        Input image.
    mode: {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval: float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    hxx, hxy, hyy: ndarray, ndim=2
        Gradients in x, y directions respectively.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional!')

    hyy = correlate1d(image, [1, -2, 1], 0, mode=mode, cval=cval)
    hxx = correlate1d(image, [1, -2, 1], 1, mode=mode, cval=cval)

    imx = correlate1d(image, [-0.5, 0, 0.5], 1, mode=mode, cval=cval)
    hxy = correlate1d(imx, [-0.5, 0, 0.5], 0, mode=mode, cval=cval)

    return hxx, hxy, hyy

# private functions


def _apply_laplacian(nx, ny, alpha, image):
    """ Apply isotropic Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of 2D image
    alpha: float
        Diffusion parameter
    image: ndarray, ndim=1
        Image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of Laplacian
    """

    # reshape
    n = len(image)
    image = image.reshape((ny, nx))

    output = image - alpha * laplace(image)
    output = output.reshape(n)

    return output


def _solve_laplacian(image, alpha=1.0, maxiter='20'):
    """ Use CG to solve a second order differential equation.

    Parameters
    ----------
    image: ndarray, ndim=2
       Input image
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    n = image.size
    (ny, nx) = image.shape

    image = image.reshape(n)
    pmatvec = partial(_apply_laplacian, nx, ny, alpha)
    A = LinearOperator((n, n), matvec=pmatvec)

    output, _ = cg(A, image, maxiter=maxiter)
    output = output.reshape((ny, nx))

    return output


def _apply_directional_laplacian(nx, ny, d11, d12, d22, alpha, image):
    """ Apply the directional Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of image.
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameters
    image: ndarray, ndim=1
        Input image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # reshape to 2D array
    n = image.size
    image = image.reshape((ny, nx))

    # compute first and second derivatives of image
    imx, imy = compute_derivatives(image)
    imxx, imxy, imyy = compute_hessian(image)

    Dx = d11 * imx + d12 * imy
    Dy = d12 * imx + d22 * imy

    # compute derivatives of spatially varying structure tensor fields.
    Dxx, _ = compute_derivatives(Dx)
    _, Dyy = compute_derivatives(Dy)

    output = image - alpha * (Dxx + Dyy)

    output = output.reshape(n)

    return output

def _imapply_directional_laplacian(image, d11, d12, d22, alpha):
    """ Apply the directional Laplacian to an image.

    Parameters
    ----------
    nx, ny: int
        Dimensions of image.
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameters
    image: ndarray, ndim=1
        Input image as a 1D vector

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # compute first and second derivatives of image
    imx, imy = compute_derivatives(image)
    imxx, imxy, imyy = compute_hessian(image)

    Dx = d11 * imx + d12 * imy
    Dy = d12 * imx + d22 * imy

    # compute derivatives of spatially varying structure tensor fields.
    Dxx, _ = compute_derivatives(Dx)
    _, Dyy = compute_derivatives(Dy)

    output = image - alpha * (Dxx + Dyy)

    return output


def _solve_directional_laplacian(image, d11, d12, d22, alpha=1.0, maxiter='20'):
    """ Use CG to solve an anisotropic second order differential equation.

    Parameters
    ----------
    image: ndarray, ndim=2
       Input image
    d11, d12, d22: ndarray, ndim=2
        Components of 2x2 structure tensor
    alpha: float
        Diffusion parameter
    maxiter: int
        Maximum number of CG iterations

    Returns
    -------
    output: ndarray, ndim=2
        Image after application of direcetional Laplacian
    """

    # get dimensions
    n = image.size
    (ny, nx) = image.shape

    # vectorize for CG
    image = image.reshape(n)

    # partial function for matrix vector product
    pmatvec = partial(_apply_directional_laplacian, nx, ny, d11, d12, d22, alpha)

    # linear operator
    A = LinearOperator((n, n), matvec=pmatvec)

    # solve linear system using CG
    output, _ = cg(A, image, maxiter=maxiter)
    output = output.reshape((ny, nx))

    return output

# smoothing functions


def apply_isotropic_smoothing(image, alpha=1.0):
    """ Smooth an image using Laplacian smoothing.

    Parameters
    ----------
    image: array_like
        Input image.
    alpha: float
        Diffusion parameter.

    Returns
    -------
    output: ndarray
        Isotropically smoothed image.
    """

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError('Image must be 2-dimensional.')

    return _solve_laplacian(image, alpha=alpha)


def apply_anisotropic_smoothing(image, d11, d12, d22, alpha=1.0):
    """ Apply anisotropic smoothing along coherent directions in image.
        Solves a directional second order PDE that is comparable to an
        anisotropic 2D diffusion equation.

    Parameters
    ----------
    image: array_like
        Input image.
    d11, d12, d22: array_like
        Diffusion tensor coefficients
    alpha: float
        Diffusion parameter.

    Returns
    -------
    output: ndarray
        Anisotropically smoothed image.
    """

    image, d11, d12, d22 = as_ndarrays(image, d11, d12, d22)
    check_2d(image, d11, d12, d22)

    return _solve_directional_laplacian(image, d11, d12, d22, alpha=alpha)

# plotting utilities


def _draw_eig_ellipse(pos, vals, vecs, scale, **kwargs):

    # Generate ellipse
    #linearity = vals[0] - vals[1] / vals[0]
    #isotropic = vals[1] / vals[0]
    linearity = np.sqrt(vals[0])
    isotropic = np.sqrt(vals[1])

    width = 0.9 * scale * linearity
    height = 0.9 * scale * isotropic

    theta = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    ellip = Ellipse(xy=pos, width=height, height=width, angle=theta, **kwargs)

    return ellip


def plot_structure_tensor(nx, nz, d11, d12, d22, dh, ax=None):

    if ax is None:
        ax = plt.gca()

    if not isinstance(dh, int):
        raise TypeError('dh should be an int')

    for i in range(0, nz, dh):
        for j in range(0, nx, dh):
            D = [[d11[i][j], d12[i][j]], [d12[i][j], d22[i][j]]]
            vals, vecs = eigsorted(D)
            ells = [_draw_eig_ellipse([j, i], vals, vecs, scale=10.0, fill=False, color='y')]

            for e in ells:
                ax.add_artist(e)

