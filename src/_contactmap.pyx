import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
DTYPE2 = np.int32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def contactmap(np.ndarray[DTYPE_t, ndim=2] frame, double cutoff):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]

    cdef np.ndarray[DTYPE2_t, ndim=2] CMAP = np.zeros((N,N), dtype=DTYPE2)
    cdef DTYPE_t tmp, dsquare
    cdef int i, j, k
    for i in xrange(N-1):
        for j in xrange(i+1, N):
             dsquare = 0.0
             for k in range(dim):
                 tmp = frame[i,k] - frame[j,k]
                 dsquare += tmp * tmp
             if dsquare <= cutoff * cutoff:
                 CMAP[j,i] = 1

    return CMAP
