
import sys
from os.path import join

import numpy as np
from obspy.core.trace import Trace

from seisflows.tools import msg, unix
from seisflows.tools.tools import exists
from seisflows.config import ParameterError
from seisflows.plugins import adjoint, misfit, readers, writers
from seisflows.tools.susignal import swindow, smute_offset, sgain_offset, sdamping

PAR = sys.modules['seisflows_parameters']
PATH = sys.modules['seisflows_paths']


class pewf2d(object):
    """ Data preprocessing class
    """

    def check(self):

        if 'MISFIT' not in PAR:
            setattr(PAR, 'MISFIT', 'Waveform')

        if 'CHANNELS' not in PAR:
            raise ParameterError(PAR, 'CHANNELS')

        if 'READER' not in PAR:
          raise ParameterError(PAR, 'READER')

        if 'WRITER' not in PAR:
          setattr(PAR, 'WRITER', PAR.READER)

        if 'NORMALIZE' not in PAR:
          setattr(PAR, 'NORMALIZE', True)

        # mute settings
        if 'MUTE' not in PAR:
          setattr(PAR, 'MUTE', False)

        if 'MUTESLOPE' not in PAR:
          setattr(PAR, 'MUTESLOPE', 0.)

        if 'MUTECONST' not in PAR:
          setattr(PAR, 'MUTECONST', 0.)

        # filter settings
        if 'PREFILTER' not in PAR:
            setattr(PAR, 'PREFILTER', False)

        if 'ZEROPHASE' not in PAR:
            setattr(PAR, 'ZEROPHASE', False)

        if 'CORNERS' not in PAR:
            setattr(PAR, 'CORNERS', 4)

        if 'BANDPASS' not in PAR:
          setattr(PAR, 'BANDPASS', False)

        if 'FREQLO' not in PAR:
          setattr(PAR, 'FREQLO', 0.)

        if 'FREQHI' not in PAR:
          setattr(PAR, 'FREQHI', 0.)

        # assertions
        if PAR.READER not in dir(readers):
            print msg.ReaderError
            raise ParameterError()

        if PAR.WRITER not in dir(writers):
            print msg.WriterError
            raise ParameterError()

        if PAR.READER != PAR.WRITER:
            print msg.DataFormatWarning % (PAR.READER, PAR.WRITER)

        if 'USE_STF_FILE' not in PAR:
            setattr(PAR, 'USE_STF_FILE', False)

        if 'STF_FILE' not in PAR:
            setattr(PAR, 'STF_FILE', 'stf.txt')

        if PAR.USE_STF_FILE:
            if not exists(join(PATH.SOLVER_INPUT, PAR.STF_FILE)):
                raise IOError('Source time function file not found.')

        if 'DAMPING' not in PAR:
            setattr(PAR, 'DAMPING', None)

        if 'GAIN' not in PAR:
            setattr(PAR, 'GAIN', None)

        if 'MUTE_OFFSET' not in PAR:
            setattr(PAR, 'MUTE_OFFSET', None)

        if PAR.MUTE_OFFSET:
            if 'MAX_OFFSET' not in PAR:
                raise ParameterError(PAR, 'MAX_OFFSET')

            if 'INNER_MUTE' not in PAR:
                setattr(PAR, 'INNER_MUTE', False)

        if 'MUTE_WINDOW' not in PAR:
            setattr(PAR, 'MUTE_WINDOW', None)

        if PAR.MUTE_WINDOW:
            if 'WINDOW' not in PAR:
                setattr(PAR, 'WINDOW', 'tukey')

            if 'TMIN' not in PAR:
                setattr(PAR, 'TMIN', 0.0)

            if 'TMAX' not in PAR:
                raise ParameterError(PAR, 'TMAX')

    def setup(self):

        # define misfit function and adjoint trace generator
        self.misfit = getattr(misfit, PAR.MISFIT)
        self.adjoint = getattr(adjoint, PAR.MISFIT)

        # define seismic data reader and writer
        self.reader = getattr(readers, PAR.READER)
        self.writer = getattr(writers, PAR.WRITER)

        # prepare channels list
        self.channels = []
        for char in PAR.CHANNELS:
            self.channels += [char]

        if PAR.USE_STF_FILE:
            stf_file = join(PATH.SOLVER_INPUT, PAR.STF_FILE)
            self.filter_stf(stf_file)


    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path+'/'+'traces/syn', channel)

            obs = self.process_traces(obs, filter=not PAR.PREFILTER)
            syn = self.process_traces(syn, filter=False)

            self.write_residuals(path, syn, obs)
            self.store_residuals(path, channel, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, channel)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Prepares solver for gradient evaluation by writing residuals and
            adjoint traces
        """
        if exists(path + '/residuals'):
            unix.rm(path + '/residuals')

        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path_try+'/'+'traces/syn', channel)

            obs = self.process_traces(obs, filter=not PAR.PREFILTER)
            syn = self.process_traces(syn, filter=False)

            self.write_residuals(path_try, syn, obs)


    def process_traces(self, stream, filter=True):
        """ Performs data processing operations on traces
        """
        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)
        df = dt**-1

        # offset mute
        if PAR.MUTE_OFFSET:
            stream = smute_offset(stream, PAR.MAX_OFFSET * 1000, inner_mute=PAR.INNER_MUTE)

        # time windowing
        if PAR.MUTE_WINDOW:
            stream = swindow(stream, tmin=PAR.TMIN, tmax=PAR.TMAX, wtype=PAR.WINDOW, units='time')

        if PAR.DAMPING > 0:
            stream = sdamping(stream, twin=PAR.DAMPING)

        # offset gain
        if PAR.GAIN:
            stream = sgain_offset(stream)

        # Filtering
        if filter:
            for trace in stream:
                if PAR.FREQLO and PAR.FREQHI:
                    trace.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=PAR.CORNERS, zerophase=PAR.ZEROPHASE)
                elif PAR.FREQHI:
                    trace.filter('lowpass', freq=PAR.FREQHI, corners=PAR.CORNERS, zerophase=PAR.ZEROPHASE)
                else:
                    pass
                    #raise ParameterError(PAR, 'BANDPASS')

        stream = self.convert_to_float(stream)

        return stream

    def write_residuals(self, path, syn, dat):
        """ Computes residuals from observations and synthetics
        """
        nt, dt, _ = self.get_time_scheme(syn)
        nn, _ = self.get_network_size(syn)

        filename = path +'/'+ 'residuals'
        if exists(filename):
            rsd = list(np.loadtxt(filename))
        else:
            rsd = []

        for ii in range(nn):
            rsd.append(self.misfit(syn[ii].data, dat[ii].data, nt, dt))

        np.savetxt(filename, rsd)

    def store_residuals(self, path, channel, s, d):

        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        filename = path +'/U{}_res.su'.format(channel)
        r = s.copy()
        for i in range(n):
            r[i].data[:] -= d[i].data[:]

        self.convert_to_float(r)
        r.write(filename, format='SU')


    def write_adjoint_traces(self, path, s, d, channel):
        """ Generates adjoint traces from observed and synthetic traces
        """
        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        for i in range(n):
            s[i].data = self.adjoint(s[i].data, d[i].data, nt, dt)

        self.writer(s, path, channel, tag='adj')

    ### utility functions

    def convert_to_float(self, stream):
        """ Converts data to float. Required for SU file formats.
        """

        nt, dt, _ = self.get_time_scheme(stream)
        n, _ = self.get_network_size(stream)

        for i in range(n):
            stream[i].data = stream[i].data.astype(dtype='float32')

        return stream


    # def write_zero_traces(self, path, channel):
    #     #TODO FIX implementation
    #     from obspy.core.stream import Stream
    #     from obspy.core.trace import Trace
    #
    #     # Solver parameter class
    #     p = Par()
    #     p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))
    #
    #     # construct seismic data and headers
    #     t = Trace(data=np.zeros(p.ntimesteps, dtype='float32'))
    #     t.stats.delta = p.dt
    #     s = Stream(t)*PAR.NREC
    #
    #     # write to disk
    #     self.writer(s, path, channel)


    def get_time_scheme(self, stream):
        # assumes uniformly sampled data across all traces
        dt = stream[0].stats.su.trace_header.\
             sample_interval_in_ms_for_this_trace * 1e-6
        nt = len(stream[0].data)
        t0 = 0.
        return nt, dt, t0


    def get_network_size(self, stream):
        nrec = len(stream)
        nsrc = 1
        return nrec, nsrc

    def filter_stf(self, file):

        stf_file = join(PATH.SOLVER_INPUT, PAR.STF_FILE)
        ts = np.loadtxt(stf_file)
        t = ts[:, 0]
        d = ts[:, 1]
        dt = t[1] - t[0]
        stf = Trace(d)
        stf.stats.delta = dt

        if PAR.FREQLO and PAR.FREQHI:
            stf.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=PAR.CORNERS, zerophase=PAR.ZEROPHASE)
        elif PAR.FREQHI:
            stf.filter('lowpass', freq=PAR.FREQHI, corners=PAR.CORNERS, zerophase=PAR.ZEROPHASE)
        else:
            pass
            #raise ParameterError(PAR, 'BANDPASS')

        fstf_file = join(PATH.SOLVER_INPUT, 'stf_f.txt')

        # write as ascii time series
        with open(fstf_file, 'w') as f:
            for i in range(len(d)):
                f.write('{:f} {:f}\n'.format(t[i], stf.data[i]))


