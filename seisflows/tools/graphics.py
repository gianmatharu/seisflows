import numpy as np
from os.path import join
from obspy import read
from obspy.core.stream import Stream
import matplotlib.pyplot as plt
from seisflows.plugins.solver.pewf2d import Par, event_dirname
from seisflows.tools.math import normalize


def cscale(v):
    return -abs(v).max(), abs(v).max()


def plot_vector(v, xlabel='', ylabel='', title=''):

    if v.ndim > 2:
        raise ValueError('v must be a vector or a time series')

    if v.ndim == 1:
        x = range(len(v))
        y = v
    else:
        x = v[:, 0]
        y = v[:, 1]

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


def create_im_subplot(data, ax=None, title='', clip=100):
    """ Create a subplot
    """
    if ax is None:
        ax = plt.gca()

    sp = ax.imshow(data, aspect='auto')
    ax.set_title(title)

    if clip is not None:
        perc = clip / 100.0
        vmin, vmax = -np.abs(data).max(), np.abs(data).max()
        vmin *= perc
        vmax *= perc
        sp.set_clim(vmin, vmax)

    return sp


def plot_model(model, p, smodel=None, cmap='seismic_r', clip=None):

    # check input
    if not isinstance(p, Par):
        raise TypeError('p should be of type Par')

    # get dimensions of subplot
    rows = 2 if smodel else 1
    cols = len(model)

    # prepare plot
    f, axes = plt.subplots(rows, cols, squeeze=False)
    plt.set_cmap(cmap)
    seis = []

    # reshape model
    model = _reshape_model_dict(model, p.nx, p.nz)

    for key in model.keys():
        seis.append([model[key], '{}'.format(key)])

    for i, item in enumerate(seis):
        create_im_subplot(seis[i][0], axes[0, i], title=seis[i][1], clip=clip)

    # add smooth plot
    if smodel:
        seis = []
        smodel = _reshape_model_dict(smodel, p.nx, p.nz)
        for key in smodel.keys():
            seis.append([smodel[key], '{} - smooth kernel'.format(key)])

        for i, item in enumerate(seis):
            create_im_subplot(seis[i][0], axes[1, i], title=seis[i][1], clip=clip)

    plt.tight_layout()
    plt.show()


def plot_model_comp(model1, model2, p, cmap='seismic_r'):

    # check input
    if not isinstance(p, Par):
        raise TypeError('p should be of type Par')

    parameters = model1.keys()

    # get dimensions of subplot
    rows = 3
    cols = len(model1)

    # prepare plot
    f, axes = plt.subplots(rows, cols, squeeze=False, sharex=True, sharey=True)
    plt.set_cmap(cmap)

    # reshape model
    model1 = _reshape_model_dict(model1, p.nx, p.nz)
    model2 = _reshape_model_dict(model2, p.nx, p.nz)

    for ikey, key in enumerate(parameters):

        axes[0, ikey].set_title(key)
        vmin = model1[key].min()
        vmax = model1[key].max()
        res = model2[key] - model1[key]
        rmin, rmax = cscale(res)

        # plt models
        axes[0, ikey].imshow(model1[key], vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, ikey].imshow(model2[key], vmin=vmin, vmax=vmax, aspect='auto')
        axes[2, ikey].imshow(res, vmin=rmin, vmax=rmax, aspect='auto')

    plt.tight_layout()
    plt.show()


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

        seis.append((xres, '{} - Ux - residuals'.format(eventid)))
        seis.append((zres, '{} - Uz - residuals'.format(eventid)))

    if adj:
        xadj, zadj = _read_components(join(path, 'adj'), adj=True)
        seis.append((xadj, '{} - Ux - adjoint'.format(eventid)))
        seis.append((zadj,  '{} - Uz - adjoint'.format(eventid)))

    for i, item in enumerate(seis):
        create_im_subplot(seis[i][0], axes[i], title=seis[i][1], clip=clip)

    #plt.tight_layout()
    plt.show()


def _reshape_model_dict(model, nx, nz):
    for key in model.keys():
        model[key] = model[key].reshape((nz, nx))

    return model


def _read_component(file):
    stream = read(file, dtype='float32', format='SU')
    return _convert_to_array(stream)


def _read_components(path, adj=False):

    if adj:
        xfile = join(path, 'Ux_adj.su')
        zfile = join(path, 'Uz_adj.su')
    else:
        xfile = join(path, 'Ux_data.su')
        zfile = join(path, 'Uz_data.su')

    ux = _read_component(xfile)
    uz = _read_component(zfile)

    return ux, uz


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

        for i, trace in enumerate(stream):
            output[:, i] = trace.data[:]

        return output