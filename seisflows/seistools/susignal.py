
import numpy as np
from seisflows.tools.math import nextpow2
import matplotlib.pyplot as plt

import scipy.signal as signal

def slowpass(stream, **options):
    """ Wrapper to obspy.signal.filter.lowpass for stream object.
    """

    for trace in stream:
        trace.filter('lowpass', **options)

    return stream

def sbandpass(stream, **options):
    """ Wrapper to Obspy filter. Performs bandpass filtering via obpsy.signal.filter.bandpass
    """

    for trace in stream:
        trace.filter('bandpass', **options)

    return stream


def swindow(stream, t0=0.0, tmin=None, tmax=None, units='samples', mute_interval=False,
             wtype='tukey', **kwargs):
    """ Perform time windowing on a seismic section with uniform sampling.
    """

    # get data dimensions
    nt = len(stream[0].data)

    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    # find window indicies
    if units == 'time':
        itmin = int((tmin-t0) / dt)
        itmax = int((tmax-t0) / dt)
    elif units == 'samples':
        itmin = int(tmin)
        itmax = int(tmax)
    else:
        raise ValueError('units should be samples or physical')

    win = window(nt, itmin, itmax, wtype=wtype, **kwargs)

    if mute_interval:
        win = 1 - win

    # window traces
    for trace in stream:
        trace.data[:] *= win

    return stream

def swindowc(stream, t0=0.0, tc=None, twin=None, mute_interval=False,
             wtype='tukey', **kwargs):
    """ Perform time windowing on a seismic section with uniform sampling.
    """

    # get data dimensions
    nt = len(stream[0].data)

    if not tc:
        raise ValueError('Need centre time.')
    if not twin:
        raise ValueError('Need width of time window')

    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    # find window indicies
    ic = int(tc-t0 / dt)
    winlen = int(twin / dt)

    if winlen % 2 != 0:
        winlen += 1

    itmin = int(ic - (winlen / 2))
    itmax = int(ic + (winlen / 2))

    win = window(nt, itmin, itmax, wtype=wtype, **kwargs)

    if mute_interval:
        win = 1 - win

    # window traces
    for trace in stream:
        trace.data[:] *= win

    return stream

def sdamping(stream, twin=None):
    """ Apply exponential time damping.
        Implements a crude first detection approach.
    """
    # get data dimensions
    nt = len(stream[0].data)
    dt = stream[0].stats.su.trace_header.\
            sample_interval_in_ms_for_this_trace * 1e-6

    time = np.arange(0, nt*dt, dt)

    if not twin:
        raise ValueError('Need width of time window')

    nwin = int(twin / dt)
    sigma = nwin / (2 * np.sqrt(2 * np.log(2)))

    for trace in stream:

        win = np.zeros(nt)
        # threshold for first arrival
        threshold = 1e-3 * max(abs(trace.data))

        if len(np.where(abs(trace.data) > threshold)[0]) != 0:
            tc = time[np.where(abs(trace.data) > threshold)[0][0]]

            # find window indicies
            ic = int(tc / dt)
            win = np.exp(-0.5 * (((ic - np.asarray(range(nt))) / sigma)**2))

            trace.data[:] *= win
        else:
            pass

    return stream

def gaussian_window(ic, nt):
    win = range(nt) - ic

def smute_offset(stream, offmax, inner_mute=False):
    """ Mute traces beyond a certain offset.
    stream: obspy stream
        Obspy stream object
    offmax:
        Maximum offset
    """

    if inner_mute:
        def f(x): return abs(x) <= abs(offmax)
    else:
        def f(x): return abs(x) >= abs(offmax)


    offsets = [item.stats.su.trace_header.
               distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
               for item in stream]

    offsets = np.asarray(offsets)

    # window traces
    for ir, trace in enumerate(stream):

        if f(offsets[ir]):
            trace.data[:] *= 0

    return stream

def sgain_offset(stream):
    """ Mute traces beyond a certain offset.
    stream: obspy stream
        Obspy stream object
    offmax:
        Maximum offset
    """
    offsets = [item.stats.su.trace_header.
               distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group
               for item in stream]

    offsets = np.asarray(offsets)
    offsets = offsets / float(abs(offsets).max())

    weights = offsets**2

    #weights = np.sin(abs(offsets) * 0.5 * np.pi)

    # window traces
    for ir, trace in enumerate(stream):
        trace.data[:] *= offsets[ir]**2

    return stream

def window(nt, imin=None, imax=None, wtype='tukey', **options):
    """ Generate a time window.

    Produces a zero padded window of length nt. Tapered window
    is of length imax-imin. If imin/imax are not provided they
    are set to 0/nt respectively.

    Parameters
    ----------

    nt: int
        length of window
    imin: int
        starting index of window
    imax: int
        end index of window
    type: {'cosine', 'tukey'}
        window type
    kwargs:
        kwargs for window routines

    Returns
    -------

    win: ndarray, ndim=1
        window
    """

    win = np.zeros(nt)
    # set default indicies
    if not imin:
        imin = 0

    if not imax:
        imax = nt

    # bounds check
    if imin > nt or imin < 0:
        raise ValueError('imin out of bounds.')

    if imin > nt or imin < 0:
        raise ValueError('imax out of bounds.')

    if imax <= imin:
        raise ValueError('imin must be less that imax.')

    # generate window
    nw = int(imax - imin)

    try:
        fwin = getattr(signal, wtype)
    except:
        raise AttributeError('Window type not found in scipy.signal')
    else:
        win[imin:imax] = fwin(nw, **options)

    return win

def source_time_function(t, t0, f0, stf_type='ricker', factor=1e10):

    tau = t - t0
    pi_f0_square = np.pi * np.pi * f0 * f0
    tau_square = tau * tau

    src_tf = factor * (1 - 2 * pi_f0_square * tau_square) * np.exp(-pi_f0_square * tau_square)

    return src_tf


def compute_amplitude_spectrum(t, y, Fs, pad=False, onesided=True, verbose=False):

    if pad:
        n = nextpow2(len(y))
    else:
        n = len(y)

    n = int(n)
    k = np.arange(n)
    freq = (Fs / n) * k

    # Amplitude spectrum
    Y = np.fft.fft(y, n)

    if onesided:
        freq = freq[:int(n/2)]
        Y = Y[:int(n/2)]

    if verbose:
        print('Peak frequency, fmax = {}'.format(freq[abs(Y).argmax()]))
        plt.subplot(2, 1, 1)
        plt.plot(t, y)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(freq, abs(Y))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.show()


    return freq, Y


