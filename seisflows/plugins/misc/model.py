import numpy as np


def build_homogeneous_model(nx, nz, value):
    """ Build a homogeneous 2D model

    Parameters
    ----------
    nx:
        no. cols of 2d array
    nz:
        no. rows of 2d array
    value:
        scalar value to fill array
    Returns
    -------
    output: ndarray, ndim=2
        Homogeneous model
    """
    return value * np.ones((nz, nx))


def insert_square(v, pos, width, value):
    iz, ix = pos
    v[iz-width:iz+width, ix-width:ix+width] = value
    return v


def insert_circle(v, pos, radius, value):
    iz, ix = pos
    for j in range(iz-radius, iz+radius):
        for i in range(ix-radius, ix+radius):
            r = ((j-iz)**2 + (i-ix)**2)**0.5
            if r < radius:
                v[j,i] = value

    return v

def insert_gaussian(v, pos, sigma, value):
    iz, ix = pos
    threshold = 1e-2

    width = int((-2 * sigma**2 * np.log(threshold))**0.5)

    for j in range(iz-width, iz+width):
        for i in range(ix-width, ix+width):
            r = ((j-iz)**2 + (i-ix)**2)**0.5
            mp = np.exp(-0.5 * (r / sigma)**2)

            if mp > threshold:
                v[j, i] += mp * value
    return v

def insert_rect(v, pos, height, width, value):
    iz, ix = pos
    v[iz-height:iz+height, ix-width:ix+width] = value
    return v
