import numpy as np
from obspy.core.stream import Stream
from obspy import read
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from seisflows.tools.array import readgrid

def get_tick_vectors(n, dh, space):

    dind = int(space / dh)
    ind = np.arange(0, n+1, dind)
    x = np.arange(0, (n+1) * dh, dh)
    return ind, x

def plot(file, time_series=False, xlabel='', ylabel='', title=''):
    """ Plot an ASCII file.
    """

    if time_series:
        t, data = _read_time_series(file)
    else:
        data = _read_vector(file)
        t = range(len(data))

    data = plt.plot(t, data)

    # labels
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.show()

def plot_image(file, nx=None, nz=None, dtype='float32', cmap='seismic_r'):
    """ Plot image.
    """
    # check parameters
    if nx is None or nz is None:
        raise ValueError('Need image dimensions.')

    image = readgrid(file, nx, nz, dtype=dtype)
    plt.imshow(image)
    plt.set_cmap(cmap)
    clim = (abs(image).max()) / 1
    print(image.max())
    plt.clim(-clim, clim)
    plt.show()


def plot_model(nx=None, nz=None, path='.', dtype='float32', cmap='seismic_r', mode=0):
    """ Plot velocity model.
    """

    # check parameters
    if nx is None or nz is None:
        raise ValueError('Need image dimensions.')

    if mode < 0 or mode > 2:
        raise KeyError('Mode must be 0, 1 or 2.')

    n = mode + 1
    params = []
    titles = []

    # read model
    vp = readgrid(join(path, 'vp.bin'), nx, nz, dtype=dtype)
    params.append(vp)
    titles.append('Vp')

    if mode >= 1:
        vs = readgrid(join(path, 'vs.bin'), nx, nz, dtype=dtype)
        params.append(vs)
        titles.append('Vs')

    if mode == 2:
        rho = readgrid(join(path, 'rho.bin'), nx, nz, dtype=dtype)
        params.append(rho)
        titles.append('Rho')

    # prepare plot
    plt.set_cmap(cmap)

    for i, param, title in zip(range(n), params, titles):
        plt.subplot(n, 1, i+1)
        plt.imshow(param)
        plt.title(title)

    plt.show()


def plot_section(file, format='SU', cmap='seismic_r', clip=100):
    """ Return a seismic section:
    """

    # check arguments
    get_cmap(cmap)

    # convert data to image array
    stream = read(file, format)
    im = _convert_to_array(stream)
    vmin, vmax = -np.abs(im).max(), np.abs(im).max()
    vmin *= (clip / 100.0)
    vmax *= (clip / 100.0)

    # plot section
    plt.set_cmap(cmap)
    plt.imshow(im, aspect='auto')
    plt.clim(vmin, vmax)
    plt.xlabel('Offset')
    plt.ylabel('Time (s)')

    plt.show()

def plot_obs_section(stream, cmap='seismic_r', clip=100):
    """ Return a seismic section:
    """

    # check arguments
    get_cmap(cmap)

    # convert data to image array
    im = _convert_to_array(stream)
    vmin, vmax = -np.abs(im).max(), np.abs(im).max()
    vmin *= (clip / 100.0)
    vmax *= (clip / 100.0)

    # plot section
    plt.set_cmap(cmap)
    plt.imshow(im, aspect='auto')
    plt.clim(vmin, vmax)
    plt.xlabel('Offset')
    plt.ylabel('Time (s)')



def plot_data(path, eventid, data=True, syn=False, res=False, adj=False, cmap='seismic_r', clip=100):

    # set subplot dimensions.
    rows = 2
    cols = data + syn + res + adj
    eventid = event_dirname(eventid)
    path = join(path, eventid, 'traces')

    # prepare plot
    f, axes = plt.subplots(rows, cols)
    axes = axes.transpose()
    axes = axes.reshape(rows*cols)
    plt.set_cmap(cmap)

    seis = []

    # read seismogramgs
    if data:
        xdata, zdata = _read_components(join(path, 'obs'))
        seis.append((xdata, '{} - Ux - data'.format(eventid)))
        seis.append((zdata, '{} - Uz - data '.format(eventid)))

    if syn:
        xsyn, zsyn = _read_components(join(path, 'syn'))
        seis.append((xsyn, '{} - Ux - synthetic'.format(eventid)))
        seis.append((zsyn,  '{} - Uz - synthetic'.format(eventid)))

    if res:
        if not data:
            xdata, zdata = _read_components(join(path, 'obs'))

        if not syn:
            xsyn, zsyn = _read_components(join(path, 'syn'))

        xres = xsyn - xdata
        zres = zsyn - zdata

        seis.append((xres, '{} - Ux - residua ls'.format(eventid)))
        seis.append((zres, '{} - Uz - residuals'.format(eventid)))

    if adj:
        xadj, zadj = _read_components(join(path, 'adj'), adj=True)
        seis.append((xadj, '{} - Ux - adjoint'.format(eventid)))
        seis.append((zadj,  '{} - Uz - adjoint'.format(eventid)))


    for i, item in enumerate(seis):
        create_im_subplot(seis[i][0], axes[i], title=seis[i][1], clip=clip)

    plt.show()


def plot_grad(path, nx=None, nz=None, alpha=True, beta=True, smooth=True,
              cmap='seismic_r', clip=100):

    rows = alpha + (alpha * smooth) + beta + (beta * smooth)
    cols = 1

    # prepare plot
    f, axes = plt.subplots(rows, cols)
    plt.set_cmap(cmap)

    seis = []

    # read seismogramgs
    if alpha:
        vp = readgrid(join(path, 'vp_kernel.bin'), nx, nz, dtype='float32')
        seis.append((vp, 'Vp - kernel'))

        if smooth:
            vps = readgrid(join(path, 'vp_smooth_kernel.bin'), nx, nz, dtype='float32')
            seis.append((vps, 'Vp - Smooth kernel'))

    if beta:
        vs = readgrid(join(path, 'vs_kernel.bin'), nx, nz, dtype='float32')
        seis.append((vs, 'Vs - kernel'))

        if smooth:
            vss = readgrid(join(path, 'vs_smooth_kernel.bin'), nx, nz, dtype='float32')
            seis.append((vss, 'Vs - Smooth kernel'))

    for i, item in enumerate(seis):

        create_im_subplot(seis[i][0], axes[i], title=seis[i][1], clip=clip)

    plt.show()


def plot_ev_grad(path, eventid, nx=None, nz=None, alpha=True, beta=True, rho=False, precond=True,
                 cmap='seismic_r', clip=100):

    rows = alpha + beta + rho + precond * (1 + alpha + beta)
    cols = 1
    eventid = event_dirname(eventid)
    path = join(path, eventid, 'traces', 'syn')

    # prepare plot
    f, axes = plt.subplots(rows, cols)
    plt.set_cmap(cmap)

    seis = []

    # read seismogramgs
    if alpha:
        vp = readgrid(join(path, 'vp_kernel.bin'), nx, nz, dtype='float32')
        seis.append((vp, '{} - Vp - kernel'.format(eventid)))
    if beta:
        vs = readgrid(join(path, 'vs_kernel.bin'), nx, nz, dtype='float32')
        seis.append((vs, '{} - Vs - kernel'.format(eventid)))
    if rho:
        rho = readgrid(join(path, 'rho_kernel.bin'), nx, nz, dtype='float32')
        seis.append((rho, '{} - rho - kernel'.format(eventid)))
    if precond:
        p = readgrid(join(path, 'precond.bin'), nx, nz, dtype='float32')
        p = p / abs(p).max()
        p = 1 / (p + 1e-3)
        p = p / abs(p).max()
        seis.append((p, 'Precond'.format(eventid)))

        if alpha:
            vpk = p * vp
            seis.append((vpk, 'Vp - kernel precond'.format(eventid)))
        if beta:
            vsk = p * vs
            seis.append((vsk, 'Vs-kernel precond'.format(eventid)))


    for i, item in enumerate(seis):
        create_im_subplot(seis[i][0], axes[i], title=seis[i][1], clip=clip/100)

    plt.show()

def check_opt(filename, nx=None, nz=None, title='', materials='Elastic', cmap='seismic_r', clim='mirror'):

    parameters = []
    if materials == 'Elastic':
        parameters += ['vp']
        parameters += ['vs']
    elif materials == 'Acoustic':
        parameters += ['vp']

    npar = len(parameters)
    seis = []

    # load numpy file
    nv = np.load(filename)
    n = int(len(nv) / npar)

    # split and reshape vectors
    for ipar, par in enumerate(parameters):

        v = nv[(ipar*n):(ipar*n) + n]
        v = v.reshape((nz, nx))
        seis.append((v, '{} - {}'.format(par, title)))

    # prepare plots
    f, axes = plt.subplots(npar+1, 1)
    plt.set_cmap(cmap)

    for i, item in enumerate(seis):
        create_im_subplot(seis[i][0], axes[i], title=seis[i][1], clim=clim)

    plt.show()


def create_im_subplot(data, ax=None, title='', clip=100, clim='mirror'):
    """ Create a subplot
    """
    if ax is None:
        ax = plt.gca()

    perc = clip / 100
    vmin, vmax = -np.abs(data).max(), np.abs(data).max()
    vmin *= perc
    vmax *= perc
    sp = ax.imshow(data, aspect='auto')
    ax.set_title(title)

    if clim == 'mirror':
        sp.set_clim(vmin, vmax)

    return sp

def compare_residuals(dpath, spath, cmap='seismic_r', clip=100):

    # read data
    xdata, zdata = _read_components(dpath)
    xsyn, zsyn = _read_components(spath)

    xres = xsyn - xdata
    zres = zsyn - zdata

    plt.set_cmap(cmap)

    # x components
    vmin, vmax = -np.abs(xdata).max(), np.abs(xdata).max()
    vmin *= (clip / 100.0)
    vmax *= (clip / 100.0)
    plt.subplot(2, 3, 1)
    plt.imshow(xdata, aspect='auto')
    plt.clim(vmin, vmax)
    plt.subplot(2, 3, 2)
    plt.imshow(xsyn, aspect='auto')
    plt.clim(vmin, vmax)
    plt.subplot(2, 3, 3)
    plt.imshow(xres * 1, aspect='auto')
    plt.clim(vmin, vmax)

    # z components
    vmin, vmax = -np.abs(zdata).max(), np.abs(zdata).max()
    vmin *= (clip / 100.0)
    vmax *= (clip / 100.0)
    plt.subplot(2, 3, 4)
    plt.imshow(zdata, aspect='auto')
    plt.clim(vmin, vmax)
    plt.subplot(2, 3, 5)
    plt.imshow(zsyn, aspect='auto')
    plt.clim(vmin, vmax)
    plt.subplot(2, 3, 6)
    plt.imshow(zres * 1, aspect='auto')
    plt.clim(vmin, vmax)

    plt.show()

def _read_components(path, adj=False):

    if adj:
        xfile = join(path, 'Ux_adj.su')
        zfile = join(path, 'Uz_adj.su')
    else:
        xfile = join(path, 'Ux_data.su')
        zfile = join(path, 'Uz_data.su')

    xstream = read(xfile, dtype='float32', format='SU')
    zstream = read(zfile, dtype='float32', format='SU')
    ux = _convert_to_array(xstream)
    uz = _convert_to_array(zstream)

    return ux, uz


def _read_vector(file):
    A = np.loadtxt(file)
    return A[:]


def _read_time_series(file):
    A = np.loadtxt(file)
    return A[:, 0], A[:, 1]


def _convert_to_array(stream):
    """ Extract trace data from an Obspy stream and return a 2D array

    Parameters
    ----------
    stream: Obspy stream object
        Stream storing trace data.

    Returns
    -------
    output: ndarray, ndim=2
        Returns an (nt*nr) array. nt and nr are the number of sample points
        and receivers respectively. Each column stores trace data for a single
        receiver. Assumes trace lengths are equal for all traces.
    """

    try:
        isinstance(stream, Stream)
    except:
        raise TypeError('Input object should be an Obspy stream')
    else:
        nt = len(stream.traces[0].data)
        nr = len(stream)
        output = np.zeros((nt, nr))
        i = 0
        for trace in stream:
            output[:, i] = trace.data[:]
            i += 1

        return output


def event_dirname(n):
    """ return string with event directory name
    """
    return '{:03d}'.format(n)
