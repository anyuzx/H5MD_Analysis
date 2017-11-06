"""
ps.pyx
Contact Probability Profile
Calculate the contact probability between
two beads on a polymer with linear separation
s. 
return the array with first column be separation s,
second colum to be the mean value of contact probability

P(s) = (1/(N-s))*\sum_{i=1}^{N-s}(\delta(r_{i+1} - r_{i}<cutoff))
"""

import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def ps(np.ndarray[DTYPE_t, ndim=2] frame, double cutoff):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] ps = np.zeros((N-1,2), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i, j, k
    for i in xrange(N-1):
        for j in xrange(i+1, N):
             dsquare = 0.0
             for k in range(dim):
                 tmp = frame[i,k] - frame[j,k]
                 dsquare += tmp * tmp
             if dsquare <= cutoff * cutoff:
                 ps[j-i-1,1] += 1.0/(N-(j-i))

    tmp = 1.0
    for i in xrange(N-1):
        ps[i,0] = tmp
        tmp += 1.0

    return ps
