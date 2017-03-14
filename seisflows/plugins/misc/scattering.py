import numpy as np
import matplotlib.pyplot as plt


def _check_scattering_dict(W, parameters):
    """ Check input.
    """

    diff = set(parameters) - set(W.keys())
    if diff:
        raise KeyError('W is missing keys {}'.format(diff))

    for key in parameters:
        if len(W[key]) != 3:
            raise ValueError('{} scattering modes incomplete.'.format(key))


def plot_rad_pats(W, parameters, plot_modes=False):
    """ Plot scattering patterns for PP, PS (=SP) and SS modes.

    Parameters
    ----------
    W: Dictionary of function lists.
        Dictionary containing scattering patterns. Each item
        should contain a function list of length 3 containing
        radiation patterns for PP, PS and SS scattering modes.
    """

    # check input
    _check_scattering_dict(W, parameters)
    modes = ['PP', 'PS', 'SS']

    # implicit ordering is PP PS SS
    if plot_modes:
        n = len(modes)
        cdict = dict(zip(parameters, ['r', 'b', 'k']))
    else:
        n = len(parameters)
        cdict = dict(zip(modes, ['r', 'b--', 'k--']))

    theta = 2 * np.pi * np.arange(0, 1, 0.005)

    # plot
    fig = plt.subplots(figsize=(15, 10))

    if plot_modes:
        outer_list = modes
        inner_list = parameters
    else:
        outer_list = parameters
        inner_list = modes

    for i, outer_item in enumerate(outer_list):

        ax = plt.subplot(1, n, i+1, projection='polar')
        ax.set_theta_zero_location("N")

        for j, inner_item in enumerate(inner_list):

            if plot_modes:
                rfunc = W[inner_item][i]
            else:
                rfunc = W[outer_item][j]

            if callable(rfunc):
                rpat = rfunc(theta)
            else:
                rpat = np.ones(theta.shape) * rfunc

            maxval = abs(rpat).max()
            if maxval == 0:
                ax.plot(theta, rpat, cdict[inner_item], label=inner_item)
            else:
                ax.plot(theta, abs(rpat) / maxval, cdict[inner_item], label=inner_item)

            # set labels
            ax.set_rticks([1, 2])
            ax.set_xticks(np.arange(0, 2*np.pi, np.pi / 2))
            ax.set_title(outer_item)

    plt.legend(shadow=True, fancybox=True)
    plt.tight_layout()


def lame(vp, vs, rho):
    """ Scattering modes for Lame parameters (PP, PS/SP, SS)
    Returns
    -------
    W: dict of function lists
        Returns dictionary. Each dictionary element contains scattering functions
        for a perturbation in the key parameter.
    """
    W = {}
    W['rho'] = [np.cos, np.sin, np.cos]
    W['lambda'] = [(1 / vp**2), 0,  0]
    W['mu'] = ([lambda theta: (2 / vp**2) * np.cos(theta) * np.cos(theta),
                lambda theta: (1 / (vp*vs)) * np.sin(2*theta),
                lambda theta: (1 / vs**2) * np.cos(2*theta)])

    return W


def velocity(vp, vs, rho):
    """ Scattering modes for Lame parameters (PP, PS/SP, SS)
    Returns
    -------
    W: dict of function lists
        Returns dictionary. Each dictionary element contains scattering functions
        for a perturbation in the key parameter.
    """
    W = {}
    # density scattering
    W['rho'] = [lambda theta: 1 + np.cos(theta) + (vs**2/vp**2) * (np.cos(2*theta) - 1),
                lambda theta: np.sin(theta) + (vs/vp) * np.sin(2*theta),
                lambda theta: np.cos(theta) + np.cos(2*theta)]
    W['vp'] = [(2 * rho) / vp, 0, 0]
    W['vs'] = [lambda theta: ((2 * rho * vs) / vp**2) * (np.cos(2*theta) - 1),
               lambda theta: ((2 * rho) / vp) * np.sin(2*theta),
               lambda theta: ((2 * rho) / vs) * np.cos(2*theta)]

    return W


def impedance(vp, vs, rho):
    """ Scattering modes for Lame parameters (PP, PS/SP, SS)
    Returns
    -------
    W: dict of function lists
        Returns dictionary. Each dictionary element contains scattering functions
        for a perturbation in the key parameter.
    """
    W = {}
    W['rho'] = [lambda theta: -1 + np.cos(theta) - (vs**2/vp**2) * (np.cos(2*theta)-1),
                lambda theta: np.sin(theta) - (vs/vp) * np.sin(2*theta),
                lambda theta: np.cos(theta) - np.cos(2*theta)]
    W['Ip'] = [2 / vp, 0, 0]
    W['Is'] = [lambda theta: ((2 * vs) / vp**2) * (np.cos(2*theta) - 1),
               lambda theta: (2 / vp) * np.sin(2*theta),
               lambda theta: (2 / vs) * np.cos(2*theta)]

    return W


def slowness(vp, vs, rho):
    """ Scattering modes for Lame parameters (PP, PS/SP, SS)
    Returns
    -------
    W: dict of function lists
        Returns dictionary. Each dictionary element contains scattering functions
        for a perturbation in the key parameter.
    """
    W = {}
    W['rho'] = [lambda theta: 1 + np.cos(theta) + (vs**2/vp**2) * (np.cos(2*theta) - 1),
                lambda theta: np.sin(theta) + (vs/vp) * np.sin(2*theta),
                lambda theta: np.cos(theta) + np.cos(2*theta)]
    W['pp'] = [-(2 * rho * vp), 0, 0]
    W['ps'] = [lambda theta: (-(2 * rho * vs**3) / vp**2) * (np.cos(2*theta) - 1),
               lambda theta: (-(2 * rho * vs**2) / vp) * np.sin(2*theta),
               lambda theta: (-(2 * rho * vs)) * np.cos(2*theta)]

    return W


def bulk(vp, vs, rho):
    """ Scattering modes for Lame parameters (PP, PS/SP, SS)
    Returns
    -------
    W: dict of function lists
        Returns dictionary. Each dictionary element contains scattering functions
        for a perturbation in the key parameter.
    """
    mu = rho * vs * vs
    lam = rho*vp**2 - rho * vs**2
    bulk = lam + (2 / 3) * mu
    bss = np.sqrt(bulk / rho)
    W = {}
    W['rho'] = [lambda theta: np.cos(theta) + (1/vp**2) * (bss**2 - (2/3)*vs**2) +
                              2*(vs**2/vp**2) * np.cos(theta) * np.cos(theta),
                lambda theta: np.sin(theta) + (vs/vp) * np.sin(2*theta),
                lambda theta: np.cos(theta) + np.cos(2*theta)]
    W['vb'] = [(2 * rho * bss) / vp**2, 0, 0]
    W['vs'] = [lambda theta: ((4 * rho * vs) / vp **2) * (np.cos(theta) * np.cos(theta) - (1/3)),
               lambda theta: ((2 * rho) / vp) * np.sin(2*theta),
               lambda theta: ((2 * rho) / vs) * np.cos(2*theta)]

    return W
