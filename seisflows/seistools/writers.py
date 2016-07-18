
import numpy as np


def su(d, path, filename):
    max_delta = 0.065535
    dummy_delta = max_delta

    if d[0].stats.delta > max_delta:
        for t in d:
            t.stats.delta = dummy_delta

    # write data to file
    d.write(path+'/'+filename, format='SU')


def ascii(stream, path, filenames):
    for ir, tr in enumerate(stream):
        nt = tr.stats.npts
        t1 = float(tr.stats.starttime)
        t2 = t1 + tr.stats.npts*tr.stats.sampling_rate
        print nt, t1, t2

        t = np.linspace(t1, t2, nt)
        w = tr.data

        #print path +'/'+ tr.stats.filename
        #print t.shape, tr.data.shape
        np.savetxt(path +'/'+ tr.stats.filename,
                   np.column_stack((t, w)))


def su_ewf2d_obspy(d, prefix='', channel=None, tag='data'):
    """ Write Seismic Unix file
    """

    if channel in ['x']:
        file = '%s/Ux_%s.su' % (prefix, tag)
    elif channel in ['z']:
        file = '%s/Uz_%s.su' % (prefix, tag)
    else:
        raise ValueError('CHANNEL must be one of the following: x z')

    # write data to file
    d.write(file, format='SU', byteorder='<')
