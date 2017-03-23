import numpy as np

def Hz2ERBnum(Hz):
    return 21.4*np.log10(Hz*0.00437 + 1.0)

def ERBnum2Hz(ERB):
    return (10**(ERB/21.4)-1)/4.37e-3

class gtfb:
    """Superclass for gammatone filterbank objects."""
    
    def __init__(self, fs=16000, frange=[80, 5000], nfilt=32, cfs=None):

        if cfs is None:
            ERBlo = Hz2ERBnum(frange[0])
            ERBhi = Hz2ERBnum(frange[1])
            cfs = ERBnum2Hz(np.linspace(ERBlo, ERBhi, nfilt))

        self.cfs = cfs
        self.ERB = 0.1079*cfs+24.7
        self.nfilt = cfs.shape[0]
        self.fs = fs

        # shortcuts used by derived methods
        self._omega = 2*np.pi*cfs/fs
        self._normB = 2*np.pi*1.019*self.ERB/fs

