from seisflows.tools.array import readgrid
from os.path import join
import matplotlib.pyplot as plt
from seisflows.seistools.ewf2d import read_models
import numpy as np


g_path = '/home/gian/Desktop/seisflows/examples/PSF/unperturbed/evalgrad'
mb_path = '/home/gian/Desktop/seisflows/examples/PSF/models/model_init'
mp_path = '/home/gian/Desktop/seisflows/examples/PSF/models/model_pert'

def hessian_kernel_c(nx, nz, path_grad, path_model_init, path_model_pert, display=False):

    pars = ['rho', 'vp', 'vs']

    # read initial and perturbed model
    rho_init, vp_init, vs_init = read_models(pars, path_model_init, nx, nz)
    rho_pert, vp_pert, vs_pert = read_models(pars, path_model_pert, nx, nz)

    # compute model perturbations
    rho_diff = rho_pert - rho_init
    vp_diff = vp_pert - vp_init
    vs_diff = vs_pert - vs_init

    # read gradients computed in background model
    rhok, vpk, vsk = read_models(pars, path_grad, nx, nz, kernels=True)

    K1 = 0 * rho_diff + (vpk / rho_init) * vp_diff + (vsk / rho_init) * vs_diff
    K2 = (vpk / rho_init) * rho_diff + (vpk/vp_init) * vp_diff + 0 * vs_diff
    K3 = (vsk / rho_init) * rho_diff + 0 * vp_diff + (vsk / vs_init) * vs_diff

    if display:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(K1)
        plt.clim(-abs(K1).max(), abs(K1).max())
        plt.subplot(1, 3, 2)
        plt.imshow(K2)
        plt.clim(-abs(K2).max(), abs(K2).max())
        plt.subplot(1, 3, 3)
        plt.imshow(K3)
        plt.clim(-abs(K3).max(), abs(K3).max())

    K1.astype('float32').tofile('Kc_1.bin')
    K2.astype('float32').tofile('Kc_2.bin')
    K3.astype('float32').tofile('Kc_3.bin')
