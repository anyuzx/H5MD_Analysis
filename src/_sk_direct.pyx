# cython: language_level=3
# cython: linetrace=True
import cython
import numpy as np
cimport numpy as np

from libc.math cimport sqrt
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport pow
from libc.math cimport M_PI
from libc.math cimport floor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def sk_direct_self(double [:,:] positions not None, double kmax, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n = positions.shape[0]

    cdef double dkx = 2. * M_PI / l[0]
    cdef double dky = 2. * M_PI / l[1]
    cdef double dkz = 2. * M_PI / l[2]

    cdef Py_ssize_t nkx = int(floor(kmax / dkx))
    cdef Py_ssize_t nky = int(floor(kmax / dky))
    cdef Py_ssize_t nkz = int(floor(kmax / dkz))

    cdef Py_ssize_t nk = nkx * nky * nkz

    sk = np.zeros((nk, 2), dtype = np.float64)
    cdef double [:, :] sk_view = sk

    cosk = np.zeros(nk, dtype = np.float64)
    cdef double [:] cosk_view = cosk

    sink = np.zeros(nk, dtype = np.float64)
    cdef double [:] sink_view = sink

    cdef double k_magnitude, kr_product, kx, ky, kz
    cdef Py_ssize_t ikx, iky, ikz, i, q

    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                q = ikx * nky * nkz + iky * nkz + ikz
                kx = ikx * dkx
                ky = iky * dky
                kz = ikz * dkz
                k_magnitude = sqrt(pow(kx, 2.) + pow(ky, 2.) + pow(kz, 2.))
                for i in range(n):
                    kr_product = kx * positions[i,0] + \
                                 ky * positions[i,1] + \
                                 kz * positions[i,2]
                    cosk_view[q] += cos(kr_product)
                    sink_view[q] += sin(kr_product)
                sk_view[q,0] = k_magnitude
                sk_view[q,1] = (1.0 / n) * (pow(cosk_view[q], 2.) + pow(sink_view[q], 2.))
    
    return sk


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) # used to not checking division, can have 20% performance speed up
def sk_direct_cross(double [:,:] positions1 not None, double [:,:] positions2 not None, double kmax, double [:] l not None):
    # This function compute the pair-wise distances in a orthorhombic box with sides of length l1, l2 and l3
    cdef Py_ssize_t n1 = positions1.shape[0]
    cdef Py_ssize_t n2 = positions2.shape[0]

    cdef double dkx = 2. * M_PI / l[0]
    cdef double dky = 2. * M_PI / l[1]
    cdef double dkz = 2. * M_PI / l[2]

    cdef Py_ssize_t nkx = int(floor(kmax / dkx))
    cdef Py_ssize_t nky = int(floor(kmax / dky))
    cdef Py_ssize_t nkz = int(floor(kmax / dkz))

    cdef Py_ssize_t nk = nkx * nky * nkz

    sk = np.zeros((nk, 2), dtype = np.float64)
    cdef double [:, :] sk_view = sk

    cosk1 = np.zeros(nk, dtype = np.float64)
    cdef double [:] cosk1_view = cosk1

    cosk2 = np.zeros(nk, dtype = np.float64)
    cdef double [:] cosk2_view = cosk2

    sink1 = np.zeros(nk, dtype = np.float64)
    cdef double [:] sink1_view = sink1

    sink2 = np.zeros(nk, dtype = np.float64)
    cdef double [:] sink2_view = sink2

    cdef double k_magnitude, kr_product, kx, ky, kz
    cdef Py_ssize_t ikx, iky, ikz, i, q

    for ikx in range(nkx):
        for iky in range(nky):
            for ikz in range(nkz):
                q = ikx * nky * nkz + iky * nkz + ikz
                kx = ikx * dkx
                ky = iky * dky
                kz = ikz * dkz
                k_magnitude = sqrt(pow(kx, 2.) + pow(ky, 2.) + pow(kz, 2.))
                for i in range(n1):
                    kr_product = kx * positions1[i,0] + \
                                 ky * positions1[i,1] + \
                                 kz * positions1[i,2]
                    cosk1_view[q] += cos(kr_product)
                    sink1_view[q] += sin(kr_product)

                for i in range(n2):
                    kr_product = kx * positions2[i,0] + \
                                 ky * positions2[i,1] + \
                                 kz * positions2[i,2]
                    cosk2_view[q] += cos(kr_product)
                    sink2_view[q] += sin(kr_product)

                sk_view[q,0] = k_magnitude
                sk_view[q,1] = (1.0 / (n1 + n2)) * (cosk1_view[q] * cosk2_view[q] + sink1_view[q] * sink2_view[q])
    
    return sk
