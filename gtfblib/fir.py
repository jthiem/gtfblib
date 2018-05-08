from __future__ import division
import numpy as np

from .gtfb import gtfb
from .olafilt import olafilt

class FIR(gtfb):

    def __init__(self, complexresponse=False, L=None, reversetime=False,
                 groupdelay=None, **kwargs):
        """Initialize FIR gammatone filterbank coefficients."""
        gtfb.__init__(self, **kwargs)

        # if not given, calculate the length of the response
        # should be based on lowest frequency filter
        # for now, just 100ms long
        if L is None:
            L = int(np.ceil(self.fs*0.1))
        self.L = L

        # handle group delay option:
        #  if set to 0, group delay is different for every filter and
        #   minimal for that filter (not aligned)
        #  if not set (None, default), group delay is minimum set by
        #   the lowest frequency filter
        #  else it must be a positive integer: the peak and zero phase
        #   will be at that sample.
        if groupdelay is 0:
            t = np.arange(1, L+1)/self.fs
            edelay = np.zeros(self.nfilt)
        if groupdelay is None:
            groupdelay = int(np.ceil(3/self._normB[0]))
            # print('Group delay set to', groupdelay)
        if groupdelay>0:
            t = np.arange(-groupdelay+1, L-groupdelay+1)/self.fs
            edelay = 3/(self._normB * self.fs)
        self.groupdelay = groupdelay

        # compute impulse responses
        self.ir = np.zeros((L, self.cfs.shape[0]), dtype=np.complex128)
        for n, cf in enumerate(self.cfs):
            env = ((2*np.pi*self.ERB[n])**4)/6 * (t+edelay[n])**3 \
                    * np.exp(-2*np.pi*self.ERB[n]*(t+edelay[n]))
            env[env<0] = 0
            self.ir[:, n] = env * np.exp(2*np.pi*1j*self.cfs[n]*t)

        # if real result is wanted, convert now
        if not complexresponse:
            self.ir = self.ir.real

        if reversetime:
            self.ir = np.fliplr(self.ir)

        # set initial conditions
        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._memory = [np.zeros(1, dtype=self.ir.dtype)
                             for n in range(self.nfilt)]

    def process(self, insig):
        out = np.zeros((insig.shape[0], self.nfilt), dtype=self.ir.dtype)
        for n in range(self.nfilt):
            out[:, n], self._memory[n] = olafilt(self.ir[:, n], insig,
                                                 self._memory[n])
        return out

    def process_single(self, insig, n):
        return olafilt(self.ir[:, n], insig)
