#!/usr/local/bin/python3.5
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb
from .olafilt import olafilt

class FIR(gtfb):

    def __init__(self, complexresponse=False, L=None, reversetime=False, **kwargs):
        """Initialize FIR gammatone filterbank coefficients."""
        gtfb.__init__(self, **kwargs)

        # if not given, calculate the length of the response
        # should be based on lowest frequency filter
        # for now, just 100ms long
        if L is None:
            L = int(np.ceil(self.fs*0.1))
        self.L = L
        
        # compute impulse responses
        self.ir = np.zeros((L, self.cfs.shape[0]), dtype=np.complex128)
        # TODO: turn off warning about cast to real if real response desired.
        for n, cf in enumerate(self.cfs):
            self.ir[:, n] = ((1/3)*np.pi*self._normB[n]*self.fs
                             *np.arange(1, L+1)**3
                             *np.exp(-self._normB[n]*np.arange(0, L))
                             *np.exp(1j*self._omega[n]*np.arange(0, L)))
        
        # if real result is wanted, convert now
        if not complexresponse:
            self.ir = self.ir.real
            
        if reversetime:
            self.ir = np.fliplr(self.ir)
               
        # set initial conditions
        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._memory = [np.zeros(1, dtype=self.ir.dtype) for n in range(self.nfilt)]
    
    def process(self, insig):
        out = np.zeros((insig.shape[0], self.nfilt), dtype=self.ir.dtype)
        for n in range(self.nfilt):
            out[:, n], self._memory[n] = olafilt(self.ir[:, n], insig, self._memory[n])
        return out
    
    def process_single(self, insig, n):
        return olafilt(self.ir[:, n], insig)
