
from os.path import join
import numpy as np
from obspy.core.trace import Trace
from seisflows.seistools.ewf2d import Par
from seisflows.tools.code import exists
from seisflows.tools.config import SeisflowsParameters, SeisflowsPaths, \
    ParameterError, custom_import
from seisflows.seistools.susignal import swindow, smute_offset, sgain_offset, sdamping

PAR = SeisflowsParameters()
PATH = SeisflowsPaths()


class ewf2d(custom_import('preprocess', 'base')):
    """ Data preprocessing class
    """

    def check(self):

        super(ewf2d, self).check()

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

        super(ewf2d, self).setup()

        if PAR.USE_STF_FILE:
            stf_file = join(PATH.SOLVER_INPUT, PAR.STF_FILE)
            self.filter_stf(stf_file)


    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path+'/'+'traces/syn', channel)

            obs = self.process_traces(obs)
            syn = self.process_traces(syn, filter=False)

            self.write_residuals(path, syn, obs)
            self.store_residuals(path, channel, syn, obs)
            self.write_adjoint_traces(path+'/'+'traces/adj', syn, obs, channel)


    def evaluate_trial_step(self, path='.', path_try=''):
        """ Prepares solver for gradient evaluation by writing residuals and
            adjoint traces
        """
        for channel in self.channels:
            obs = self.reader(path+'/'+'traces/obs', channel)
            syn = self.reader(path_try+'/'+'traces/syn', channel)

            obs = self.process_traces(obs)
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

        if PAR.DAMPING:
            stream = sdamping(stream, twin=PAR.DAMPING)

        # offset gain
        if PAR.GAIN:
            stream = sgain_offset(stream)

        # Filtering
        if filter:
            for trace in stream:
                if PAR.FREQLO and PAR.FREQHI:
                    trace.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=2, zerophase=True)
                elif PAR.FREQHI:
                    trace.filter('lowpass', freq=PAR.FREQHI, corners=2, zerophase=True)
                else:
                    pass
                    #raise ParameterError(PAR, 'BANDPASS')

        stream = self.convert_to_float(stream)

        return stream

    def process_adjoint_trace(self, trace):
        """ Implements adjoint of process_traces method. Currently not used.
        """

        trace.data = trace.data[::-1]

        if PAR.FREQLO and PAR.FREQHI:
            trace.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=2)
        elif PAR.FREQHI:
            trace.filter('lowpass', freq=PAR.FREQHI, corners=2)
        else:
            pass
            #raise ParameterError(PAR, 'BANDPASS')

        # workaround obspy dtype conversion
        trace.data = trace.data[::-1].astype(np.float32)

        return trace

    def write_residuals(self, path, s, d):
        """ Computes residuals from observations and synthetics
        """
        nt, dt, _ = self.get_time_scheme(s)
        n, _ = self.get_network_size(s)

        filename = path +'/'+ 'residuals'
        r = np.zeros(n)

        for i in range(n):
            r[i] = self.misfit(s[i].data, d[i].data, nt, dt)

        np.savetxt(filename, r)

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
            #s[i] = self.process_adjoint_trace(s[i])

        # normalize traces
        if PAR.NORMALIZE:
            for ir in range(n):
                w = np.linalg.norm(d[i], ord=2)
                if w > 0: 
                    s[:,ir] /= w

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


    def write_zero_traces(self, path, channel):
        #TODO FIX implementation
        from obspy.core.stream import Stream
        from obspy.core.trace import Trace

        # Solver parameter class
        p = Par()
        p.read_par_file(join(PATH.SOLVER_INPUT, 'par_template.cfg'))

        # construct seismic data and headers
        t = Trace(data=np.zeros(p.ntimesteps, dtype='float32'))
        t.stats.delta = p.dt
        s = Stream(t)*PAR.NREC

        # write to disk
        self.writer(s, path, channel)


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
            stf.filter('bandpass', freqmin=PAR.FREQLO, freqmax=PAR.FREQHI, corners=2, zerophase=True)
        elif PAR.FREQHI:
            stf.filter('lowpass', freq=PAR.FREQHI, corners=2, zerophase=True)
        else:
            pass
            #raise ParameterError(PAR, 'BANDPASS')

        fstf_file = join(PATH.SOLVER_INPUT, 'stf_f.txt')

        # write as ascii time series
        with open(fstf_file, 'w') as f:
            for i in range(len(d)):
                f.write('{:f} {:f}\n'.format(t[i], stf.data[i]))


