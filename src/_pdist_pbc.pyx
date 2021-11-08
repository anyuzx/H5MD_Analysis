# cython: language_level=3
import cython
import numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt
from libc.math cimport nearbyint

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def pdist_cy_omp(double [:,:] positions not None, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n = positions.shape[0]
    cdef Py_ssize_t ndim = positions.shape[1]
    
    pdistances = np.zeros(n * (n-1) // 2, dtype = np.float64)
    cdef double [:] pdistances_view = pdistances
    
    cdef double d, dd
    cdef Py_ssize_t i, j, k
    
    with nogil, parallel():
        for i in prange(n-1, schedule='dynamic'):
            for j in range(i+1, n):
                dd = 0.0
                for k in range(ndim):
                    d = positions[i,k] - positions[j,k]
                    d = d - nearbyint(d / l[k]) * l[k]
                    dd = dd + d * d
                pdistances_view[j - 1 + (2 * n - 3 - i) * i // 2] = sqrt(dd)
    
    return pdistances

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def cdist_cy_omp(double [:,:] positions1 not None, double [:,:] positions2 not None, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n1 = positions1.shape[0]
    cdef Py_ssize_t n2 = positions2.shape[0]
    cdef Py_ssize_t ndim = positions1.shape[1]
    
    pdistances = np.zeros(n1 * n2, dtype = np.float64)
    cdef double [:] pdistances_view = pdistances
    
    cdef double d, dd
    cdef Py_ssize_t i, j, k
    
    with nogil, parallel():
        for i in prange(n1, schedule='dynamic'):
            for j in range(n2):
                dd = 0.0
                for k in range(ndim):
                    d = positions1[i,k] - positions2[j,k]
                    d = d - nearbyint(d / l[k]) * l[k]
                    dd = dd + d * d
                pdistances_view[n2 * i + j] = sqrt(dd)
    
    return pdistances