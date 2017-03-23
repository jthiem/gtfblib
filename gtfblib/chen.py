#!/usr/local/bin/python3.5
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb

class Chen(gtfb):
    
    def __init__(self, N_GD=None, **kwargs):
        """Initialize coefficients for GTFB as in Chen & Hohmann 2015."""
        gtfb.__init__(self, **kwargs)
        
        # if N_GD is not set, assume value corresponding to 16 ms
        if N_GD is None:
            N_GD = np.ceil(0.016*self.fs)
            
        # filter coefficient parameter and powers
        self.Ak = np.exp(-self._normB + 1j*self._omega)
        self.Ak2 = self.Ak**2
        self.Ak3 = self.Ak**3
        # per-channel gain is based on the difference (in ERB) to the
        # next filter.  This does not work for the last filter, so we 
        # duplicate it
        ERBstep = np.diff(self.cfs)/self.ERB[:-1]
        ERBstep = np.hstack((ERBstep, ERBstep[-1]))
        self.Bk = np.sqrt(2)*ERBstep*((1-self.Ak)**4)/(self.Ak+4*self.Ak2+self.Ak3)
        # phase alignment
        N_PE = np.round(3/self._normB)
        self.Ck = np.exp(-1j*self._omega*np.fmin(N_GD, N_PE))
        # group delay
        self.Dk = np.fmax(0, N_GD-N_PE)
        
        self._clear()
        
    def _clear(self):
        """clear initial conditions"""
        self._ics0 = [lfiltic([self.Ak[n], 4*self.Ak2[n], self.Ak3[n]], 
                              [1, -2*self.Ak[n], self.Ak2[n]], np.complex128([]))
            for n in range(self.nfilt)]
        self._ics1 = [lfiltic([1,], 
                              [1, -2*self.Ak[n], self.Ak2[n]], np.complex128([]))
            for n in range(self.nfilt)]
    
    def process(self, insig):
        """Process one-dimensional signal, returning an array of signals
        which are the outputs of the filters"""
        out = np.zeros((insig.shape[0], self.nfilt), dtype=np.complex128)
        for n in range(self.nfilt):
            stage1out, self._ics0[n] = lfilter([self.Ak[n], 4*self.Ak2[n], self.Ak3[n]], 
                            [1, -2*self.Ak[n], self.Ak2[n]], insig, zi=self._ics0[n])
            stage2out, self._ics1[n] = lfilter([1,], [1, -2*self.Ak[n], self.Ak2[n]], 
                                               stage1out, zi=self._ics1[n])
            out[:, n] = self.Bk[n]*self.Ck[n]*stage2out
        return out

    def process_single(self, insig, n):
        """Process a signal with just one of the filters of the
        filterbank, without affecting the state."""
        stage1out = lfilter([self.Ak[n], 4*self.Ak2[n], self.Ak3[n]], 
                            [1, -2*self.Ak[n], self.Ak2[n]], insig)
        stage2out = lfilter([1,], [1, -2*self.Ak[n], self.Ak2[n]], stage1out)
        return self.Bk[n]*self.Ck[n]*stage2out
