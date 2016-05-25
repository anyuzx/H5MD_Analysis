"""
sdp.pyx
Subchain Distance Profile
Calculate root mean square distance between
two beads on polymer with linear separation s
return the array with first column be separation s,
second column to be the mean value of distance

R(s) = (1/(N-s))*\sum_{i=1}^{N-s}(r_{i+s} - r_{i})^{2}
"""

import numpy as np
import cython
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sdp(np.ndarray[DTYPE_t, ndim=2] frame):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] sdp = np.zeros((N-1,2), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i,j,k
    for i in xrange(N-1):
        for j in xrange(i+1, N):
             dsquare = 0.0
             for k in range(dim):
                 tmp = frame[i,k] - frame[j,k]
                 dsquare += tmp * tmp
             sdp[j-i-1,1] += dsquare/(N-(j-i))

    tmp = 1.0
    for i in xrange(N-1):
        sdp[i,0] = tmp
        sdp[i,1] = sqrt(sdp[i,1])
        tmp += 1.0

    return sdp
