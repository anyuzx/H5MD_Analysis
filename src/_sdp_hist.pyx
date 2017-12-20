"""
_sdp_hist.pyx

This module compute the distance/distance square between
pairs of beads. 

Argument:
    slist: specify the list for all the s values
s value is the linear saparation between beads.
The distance between each pair of beads are stored
instead of averageing over them. This result can
be used to compute distance distribution.
"""

import numpy as np
import cython
cimport numpy as np
from libc.math cimport sqrt

DTYPE = np.float32
DTYPE2 = np.int32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def sdp_hist(np.ndarray[DTYPE_t, ndim=2] frame, np.ndarray[DTYPE2_t, ndim=1] slist):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]
    cdef int smin = slist.min()
    cdef int smax = slist.max()
    cdef int sdim = slist.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] DLIST = np.zeros((sdim, N - smin), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i, s, k, j, jj
    jj = 0
    for i in xrange(N-1):
        j = 0
        for s in slist:
            if s <= N - i - 1:
                dsquare = 0.0
                for k in range(dim):
                    tmp = frame[i,k] - frame[i+s,k]
                    dsquare += tmp * tmp
                DLIST[j, jj] = sqrt(dsquare)
            j += 1
        jj += 1

    return DLIST

@cython.boundscheck(False)
@cython.wraparound(False)
def sdp_hist_square(np.ndarray[DTYPE_t, ndim=2] frame, np.ndarray[DTYPE2_t, ndim=1] slist):
    cdef int N = frame.shape[0]
    cdef int dim = frame.shape[1]
    cdef int smin = slist.min()
    cdef int smax = slist.max()
    cdef int sdim = slist.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] DLIST = np.zeros((sdim, N - smin), dtype=DTYPE)
    cdef DTYPE_t tmp, dsquare
    cdef int i, s, k, j, jj
    jj = 0
    for i in xrange(N-1):
        j = 0
        for s in slist:
            if s <= N - i - 1:
                dsquare = 0.0
                for k in range(dim):
                    tmp = frame[i,k] - frame[i+s,k]
                    dsquare += tmp * tmp
                DLIST[j, jj] = dsquare
            j += 1
        jj += 1

    return DLIST
