import numpy as np
import cython
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def distmap(np.ndarray[DTYPE_t, ndim=2] frame):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] DMAP = np.zeros((N,N), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i, j, k
    for i in xrange(N-1):
        for j in xrange(i+1, N):
             dsquare = 0.0
             for k in range(dim):
                 tmp = frame[i,k] - frame[j,k]
                 dsquare += tmp * tmp
             DMAP[j,i] = sqrt(dsquare)

    return DMAP

@cython.boundscheck(False)
@cython.wraparound(False)
def distmap_square(np.ndarray[DTYPE_t, ndim=2] frame):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]

    cdef np.ndarray[DTYPE_t, ndim=2] DMAP = np.zeros((N,N), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i, j, k
    for i in xrange(N-1):
        for j in xrange(i+1, N):
             dsquare = 0.0
             for k in range(dim):
                 tmp = frame[i,k] - frame[j,k]
                 dsquare += tmp * tmp
             DMAP[j,i] = dsquare

    return DMAP
