from __future__ import division
import numpy as np
from scipy.io import loadmat

from gtfblib.gtfb import ERBspacing_given_spacing
from gtfblib.chen import Chen

from test_aux_functions import peak_error

refData = loadmat('test/Chen8kTestdata.mat', squeeze_me=True)
fs = 8000
insig = refData['x']
refout = refData['x_bk_csn']

def test_Chen_setup():
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    ref_Ak = refData['C2']
    ref_Bk = refData['normalizedGain']
    ref_Ck = np.exp(refData['compensatedPhase'])
    ref_ComplexA = refData['GT_complexA']
    ref_ComplexB = refData['GT_complexB']
    assert(peak_error(fbChen.Ak, ref_Ak)<1e-10)
    assert(peak_error(fbChen.Bk, ref_Bk)<1e-10)
    assert(peak_error(fbChen.Ck, ref_Ck)<1e-10)
    assert(peak_error(fbChen.Ck, ref_Ck)<1e-10)
    assert(peak_error(fbChen.ComplexA, ref_ComplexA)<1e-10)
    assert(peak_error(fbChen.ComplexB, ref_ComplexB)<1e-10)

def test_Chen_process():
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    outsig = fbChen.process(insig)
    assert(peak_error(outsig, refout[:, :1000])<1e-10)

def test_Chen_process_single():
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    outsig = fbChen.process_single(insig, 10)
    assert(peak_error(outsig, refout[10, :1000])<1e-10)

def test_Chen_process_memory():
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    outsig1 = fbChen.process(insig[:500])
    outsig2 = fbChen.process(insig[500:])
    outsig = np.hstack((outsig1, outsig2))
    assert(peak_error(outsig, refout[:, :1000])<1e-10)

def test_Chen_clear():
    fbChen = Chen(fs=fs, cfs=ERBspacing_given_spacing(80, 0.9*4000, .5))
    _ = fbChen.process(np.random.randn(1000))
    fbChen._clear()
    outsig = fbChen.process(insig)
    assert(peak_error(outsig, refout[:, :1000])<1e-10)
