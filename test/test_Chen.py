from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_spacing
from gtfblib.chen import Chen

def test_Chen_setup():
    refData = loadmat('test/Chen8kTestdata.mat',
                      squeeze_me=True)
    ref_Ak = refData['C2']
    ref_Bk = refData['normalizedGain'] 
    ref_Ck = np.exp(refData['compensatedPhase'])
    ref_ComplexA = refData['GT_complexA']
    ref_ComplexB = refData['GT_complexB']
    fs = 8000
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    assert(np.sum(np.abs(fbChen.Ak-ref_Ak))<.001)
    assert(np.sum(np.abs(fbChen.Bk-ref_Bk))<.001)
    assert(np.sum(np.abs(fbChen.Ck-ref_Ck))<.001)
    assert(np.sum(np.abs(fbChen.Ck-ref_Ck))<.001)
    assert(np.sum(np.abs(fbChen.ComplexA-ref_ComplexA))<.001)
    assert(np.sum(np.abs(fbChen.ComplexB-ref_ComplexB))<.001)