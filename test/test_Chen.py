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
    assert(np.mean(np.abs(fbChen.Ak-ref_Ak))<1e-10)
    assert(np.mean(np.abs(fbChen.Bk-ref_Bk))<1e-10)
    assert(np.mean(np.abs(fbChen.Ck-ref_Ck))<1e-10)
    assert(np.mean(np.abs(fbChen.Ck-ref_Ck))<1e-10)
    assert(np.mean(np.abs(fbChen.ComplexA-ref_ComplexA))<1e-10)
    assert(np.mean(np.abs(fbChen.ComplexB-ref_ComplexB))<1e-10)

def test_Chen_process():
    refData = loadmat('test/Chen8kTestdata.mat',
                      squeeze_me=True)
    insig = refData['x']
    refout = refData['x_bk_csn']
    fs = 8000
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))

    outsig = fbChen.process(insig)
    assert(np.mean(np.abs(outsig-refout[:, :1000]))<1e-10)
