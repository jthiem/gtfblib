# gtfblib

A library of Gammatone Filterbank implementations

This package provides a variety of gammatone filterbank implementations, namely (references TBD):
- a pretty straight-up FIR implementation (complex or real)
- the one-zero gammatone filterbank
- Zhangli Chen's complex design
- more will be added as I have time

I try to make these implementations match MATLAB versions numerically, verified by regression tests.

## Usage

All filterbanks are implemented as classes, which take various parameters during instantiation.  The
classes implement the methods `process(indata)`, `process_single(indata, channel)`, and `_clear()`.
`process(indata)` uses and updates the filter states, so that block-based processing is possible.
if needed, the `_clear()` method can be used to clear the state back to 0.
