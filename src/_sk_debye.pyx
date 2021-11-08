# cython: language_level=3
import cython
import numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt
from libc.math cimport nearbyint
from libc.math cimport sin

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
cdef double sinc(double x) nogil:
    if x == 0.0:
        return 1.0
    else:
        return sin(x) / x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def sk_debye_self(double [:,:] positions not None, double [:] k not None, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n = positions.shape[0]
    cdef Py_ssize_t ndim = positions.shape[1]
    cdef Py_ssize_t nk = k.shape[0]

    sk = np.ones(nk, dtype = np.float64)
    cdef double [:] sk_view = sk
    #pdistances = np.zeros(n * (n-1) // 2, dtype = np.float64)
    #cdef double [:] pdistances_view = pdistances

    cdef double d, dd
    cdef Py_ssize_t i, j, p, m

    with nogil, parallel():
        for i in prange(n-1, schedule='dynamic'):
            for j in range(i+1, n):
                dd = 0.0
                for p in range(ndim):
                    d = positions[i,p] - positions[j,p]
                    d = d - nearbyint(d / l[p]) * l[p]
                    dd = dd + d * d
                for m in range(nk):
                    sk_view[m] = sk_view[m] + (2.0 / n) * sinc(k[m] * sqrt(dd))

    return sk

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def sk_debye_cross(double [:,:] positions1 not None, double [:,:] positions2 not None, double [:] k not None, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n1 = positions1.shape[0]
    cdef Py_ssize_t n2 = positions2.shape[0]
    cdef Py_ssize_t ndim = positions1.shape[1]
    cdef Py_ssize_t nk = k.shape[0]

    sk = np.zeros(nk, dtype = np.float64)
    cdef double [:] sk_view = sk
    #pdistances = np.zeros(n1 * n2, dtype = np.float64)
    #cdef double [:] pdistances_view = pdistances

    cdef double d, dd
    cdef Py_ssize_t i, j, p, m

    with nogil, parallel():
        for i in prange(n1, schedule='dynamic'):
            for j in range(n2):
                dd = 0.0
                for p in range(ndim):
                    d = positions1[i,p] - positions2[j,p]
                    d = d - nearbyint(d / l[p]) * l[p]
                    dd = dd + d * d
                for m in range(nk):
                    sk_view[m] = sk_view[m] + (1.0 / (n1 + n2)) * sinc(k[m] * sqrt(dd))

    return sk
