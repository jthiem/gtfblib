- The FIR code uses the 'plain' overlap-add code.  However, where the filters
  stay constant it is better to store the FIR filter in frequency domain, taking
  a bit more memory but saving reconverting the time domain impulse into
  frequency domain in every invocation.
  
- Slaney's 1995 gammatone filters should be implemented before I release 
  this library to 1.0.0
  