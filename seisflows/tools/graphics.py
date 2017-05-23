from os.path import join
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from obspy import read
from obspy.core.stream import Stream

from seisflows.plugins.solver.pewf2d import Par, event_dirname

def plot_gll(x, y, z):
    """ Plots values on 2D unstructured GLL mesh
    """
    r = (max(x) - min(x))/(max(y) - min(y))
    rx = r/np.sqrt(1 + r**2)
    ry = 1/np.sqrt(1 + r**2)

    f = plt.figure(figsize=(10*rx, 10*ry))
    p = plt.tricontourf(x, y, z, 125)
    plt.axis('image')
    return f, p


def plot_vector(t, v, xlabel='', ylabel='', title=''):
    """ Plots a vector or time series.
    Parameters
    ----------
    v: ndarray, ndims = 1/2
        Vector or time series to plot
    xlabel: str
        x axis label
    ylabel: str
        y axis label
    title: str
        plot title
    Raises
    ------
    ValueError
        If dimensions of v are greater than 2
    """

    # check input dimension
    if v.ndim > 2:
        raise ValueError('v must be a vector or a time series')

    if v.ndim == 1:
        x = range(len(v))
        y = v
    else:
        x = v[:, 0]
        y = v[:, 1]

    # plot
    plt.plot(t, v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_section(stream, ax=None, cmap='seismic', clip=100, title='', x_interval=1.0, y_interval=1.0):
    """ Plot a seismic section from an obspy stream.
    """
    # convert stream to image array
    data = _convert_to_array(stream)

    # get dimensions
    nr = len(stream)
    nt = len(stream[0].data)
    dt = stream[0].stats.delta
    d_aspect = nr / float(nt)

    fsize = 6
    scale_factor = 1.5

    if ax is None:
        fig, ax = plt.subplots(figsize=(fsize, scale_factor*fsize))

    im = ax.imshow(data, aspect=scale_factor*d_aspect, clim=_cscale(data, clip=clip))
    im.set_cmap(cmap)

    # labels
    ax.set_title(title)
    ax.set_xlabel('Offset [km]')
    ax.set_ylabel('Time [s]')

    #set ticks
    t = _get_time(stream)
    yticks, ytick_labels = get_regular_ticks(t, y_interval)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)

    offsets =_get_offsets(stream)
    xticks, xtick_labels = get_regular_ticks(offsets, x_interval)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    return ax


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
    f, axes = plt.subplots(rows, cols, squeeze=False, figsize=(15,5))
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
        vmin = model1[key].min() - 0.01 * model1[key].max()
        vmax = model1[key].max() + 0.01 * model1[key].max()
        res = model2[key] - model1[key]
        rmin, rmax = _cscale(res)

        # plt models
        axes[0, ikey].imshow(model1[key], vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, ikey].imshow(model2[key], vmin=vmin, vmax=vmax, aspect='auto')
        axes[2, ikey].imshow(res, vmin=rmin, vmax=rmax, aspect='auto')

    plt.tight_layout()
    plt.show()


def plot_data(path, eventid, data=True, syn=False, res=False, adj=False, cmap='seismic_r', clip=100, \
              title='', x_interval=1.0, y_interval=1.0):

    # set subplot dimensions.
    rows = 2
    cols = data + syn + res + adj

    # set path
    eventid = event_dirname(eventid)
    path = join(path, eventid, 'traces')

    # prepare plot
    section = partial(plot_section, clip=clip, x_interval=x_interval, y_interval=y_interval, cmap=cmap)
    f, axes = plt.subplots(rows, cols, squeeze=False, figsize=(15, 10))
    plt.set_cmap(cmap)

    icol = 0

    # read seismogramgs
    if data:
        xdata, zdata = _read_components(join(path, 'obs'))
        section(xdata, ax=axes[0, icol], title='Ux - data')
        section(zdata, ax=axes[1, icol], title='Uz - data')
        icol += 1
    if syn:
        xsyn, zsyn = _read_components(join(path, 'syn'))
        section(xsyn, ax=axes[0, icol], title='Ux - Synth')
        section(zsyn, ax=axes[1, icol], title='Ux - Synth')
        icol += 1
    if res:
        if not data:
            xdata, zdata = _read_components(join(path, 'obs'))

        if not syn:
            xsyn, zsyn = _read_components(join(path, 'syn'))

        xres = _compute_residuals(xsyn, xdata)
        zres = _compute_residuals(zsyn, zdata)
        section(xres, ax=axes[0, icol], title='Ux - Residuals')
        section(zres, ax=axes[1, icol], title='Uz - Residuals')
        icol += 1

    if adj:
        xadj, zadj = _read_components(join(path, 'adj'), adj=True)
        section(xadj, ax=axes[0, icol], title='Ux - Adj. source')
        section(zadj, ax=axes[1, icol], title='Uz - Adj. source')

    plt.tight_layout()
    plt.show()


def _reshape_model_dict(model, nx, nz):
    for key in model.keys():
        model[key] = model[key].reshape((nz, nx))

    return model


def _read_component(file):
    stream = read(file, dtype='float32', format='SU')
    return stream


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


def _compute_residuals(stream1, stream2):
    """ Compute residuals between two seismic sections.
    """
    res = stream1.copy()
    for tr1, tr2 in zip(res, stream2):
        tr1.data -= tr2.data

    return res


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
    if not isinstance(stream, Stream):
        raise TypeError('Input object should be an Obspy stream')

    nt = len(stream.traces[0].data)
    nr = len(stream)
    output = np.zeros((nt, nr))

    for i, trace in enumerate(stream):
        output[:, i] = trace.data[:]

    return output


def _cscale(v, clip=100):
    perc = clip / 100.
    return -perc * abs(v).max(), perc * abs(v).max()


def _get_time(stream):
    """ Get fixed time vector for stream object.
    """
    dt = stream[0].stats.delta
    nt = len(stream[0].data)
    return np.arange(0, nt*dt, dt)


def _get_offsets(stream):
    """ Return offsets.
    """
    nr = len(stream)
    offsets = np.zeros(nr)
    scalco = stream[0].stats.su.trace_header.scalar_to_be_applied_to_all_coordinates

    # set scale to km
    if scalco == 0:
        scalco = 1e-3 # assume coords are in m
    else:
        scalco = 1.0e-3 / scalco

    for i, tr in enumerate(stream):
        offsets[i] = (tr.stats.su.trace_header.group_coordinate_x - \
                      tr.stats.su.trace_header.source_coordinate_x) * scalco
    return offsets


def get_regular_ticks(v, interval):
    """ Return regular ticks.
        v must be regularly sampled.
    """
    f = interp1d(v, range(len(v)))
    begin = int(v[0] / interval) * interval
    end = v[-1]
    tick_labels = np.arange(begin, end, interval)
    ticks = f(tick_labels)

    return ticks, tick_labels
