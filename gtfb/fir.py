#!/usr/local/bin/python3.5
import numpy as np
from scipy.signal import lfilter, lfiltic

from .gtfb import gtfb

class FIR(gtfb):

    def __init__(self, complexresponse=False, L=None, **kwargs):
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
               
        # set initial conditions
        self._clear()

    def _clear(self):
        """clear initial conditions"""
        self._memory = [np.zeros(1, dtype=self.ir.dtype) for n in range(self.nfilt)]
    
    def process(self, insig):
        out = np.zeros((insig.shape[0], self.nfilt), dtype=self.ir.dtype)
        for n in range(self.nfilt):
            out[:, n], self._memory[n] = self._olafilt(self.ir[:, n], insig, self._memory[n])
        return out
    
    def process_single(self, insig, n):
        return self._olafilt(self.ir[:, n], insig)

    # overlap-add filtering
    # TODO: move conversion of b into freq. domain into __init__
    def _olafilt(self, b, x, zi=None):
        """
        Filter a one-dimensional array with an FIR filter

        Filter a data sequence, `x`, using a FIR filter given in `b`.
        Filtering uses the overlap-add method converting both `x` and `b`
        into frequency domain first.  The FFT size is determined as the
        next higher power of 2 of twice the length of `b`.

        This function is available in a stand-alone form from
        https://github.com/jthiem/overlapadd/

        Parameters
        ----------
        b : one-dimensional numpy array
            The impulse response of the filter
        x : one-dimensional numpy array
            Signal to be filtered
        zi : one-dimensional numpy array, optional
            Initial condition of the filter, but in reality just the
            runout of the previous computation.  If `zi` is None or not
            given, then zero initial state is assumed.

        Returns
        -------
        y : array
            The output of the digital filter.
        zf : array, optional
            If `zi` is None, this is not returned, otherwise, `zf` holds the
            final filter delay values.
        """

        L_I = b.shape[0]
        # Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
        L_F = 2<<(L_I-1).bit_length()  
        L_S = L_F - L_I + 1
        L_sig = x.shape[0]
        offsets = range(0, L_sig, L_S)

        # blockwise frequency domain multiplication
        if np.iscomplexobj(b) or np.iscomplexobj(x):
            FDir = np.fft.fft(b, n=L_F)
            tempresult = [np.fft.ifft(np.fft.fft(x[n:n+L_S], n=L_F)*FDir)
                          for n in offsets]
            res = np.zeros(L_sig+L_F, dtype=np.complex128)
        else:
            FDir = np.fft.rfft(b, n=L_F)
            tempresult = [np.fft.irfft(np.fft.rfft(x[n:n+L_S], n=L_F)*FDir)
                          for n in offsets]
            res = np.zeros(L_sig+L_F)

        # overlap and add
        for i, n in enumerate(offsets):
            res[n:n+L_F] += tempresult[i]

        if zi is not None:
            res[:zi.shape[0]] = res[:zi.shape[0]] + zi
            return res[:L_sig], res[L_sig:]
        else:
            return res[:L_sig]
