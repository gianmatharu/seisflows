
def su(path, filename):
    import obspy
    stream = obspy.read(path +'/'+ filename,
                   format='SU',
                   byteorder='<')
    return stream


def su_ewf2d_obspy(prefix='', channel=None):
    from obspy import read

    if channel in ['x']:
        filename = '%s/Ux_data.su' % (prefix)
    elif channel in ['z']:
        filename = '%s/Uz_data.su' % (prefix)
    else:
        raise ValueError('CHANNEL must be one of the following: x z')

    streamobj = read(filename, format='SU', byteorder='<')
    return streamobj

def ascii(path, filenames):
    from numpy import loadtxt
    from obspy.core import Stream, Stats, Trace

    stream = Stream()
    for filename in filenames:
        stats = Stats()
        data = loadtxt(path +'/'+ filename)

        stats.filename = filename
        stats.starttime = data[0,0]
        stats.sampling_rate = data[0,1] - data[0,0]
        stats.npts = len(data[:,0])

        try:
            parts = filename.split('.')
            stats.network = parts[0]
            stats.station = parts[1]
            stats.channel = temp[2]
        except:
            pass

        stream.append(Trace(data=data[:,1], header=stats))

    return stream

