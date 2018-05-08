from __future__ import division
import numpy as np

def Hz2ERBnum(Hz):
    """Return the ERB filter number for a given frequency."""
    return 21.4*np.log10(Hz*4.37e-3 + 1.0)

def ERBnum2Hz(ERB):
    """Return the frequency of given ERB filter."""
    return (10**(ERB/21.4)-1.0)/4.37e-3

def ERBspacing_given_N(cf_first, cf_last, N):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last = Hz2ERBnum(cf_last)
    cfs = ERBnum2Hz(np.linspace(ERB_first, ERB_last, N))
    return cfs

def ERBspacing_given_spacing(cf_first, cf_last, ERBstep):
    ERB_first = Hz2ERBnum(cf_first)
    ERB_last = Hz2ERBnum(cf_last)
    cfs = ERBnum2Hz(np.arange(ERB_first, ERB_last, ERBstep))
    return cfs

class gtfb:
    """Superclass for gammatone filterbank objects."""

    def __init__(self, fs=16000, cfs=None, EarQ=(1/0.108), Bfact=1.0186):

        if cfs is None:
            cfs = ERBspacing_given_N(80, 0.9*fs/2, 32)

        self.cfs = cfs
        self.ERB = cfs/EarQ+24.7
        self.nfilt = cfs.shape[0]
        self.fs = fs

        # shortcuts used by derived methods
        self._omega = 2*np.pi*cfs/fs
        self._normB = 2*np.pi*Bfact*self.ERB/fs
