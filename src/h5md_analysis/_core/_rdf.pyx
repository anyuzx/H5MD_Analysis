# cython: language_level=3
import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

from libc.math cimport sqrt
from libc.math cimport pow
from libc.math cimport nearbyint
from libc.math cimport floor

from libc.stdlib cimport abort, malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rdf_self(double [:,:] positions not None, double rmax, int bins, double [:,:] box not None):
    cdef Py_ssize_t n = positions.shape[0]
    cdef Py_ssize_t ndim = positions.shape[1]

    r_count = np.zeros(bins, dtype=np.float64)
    cdef double [:] r_count_view = r_count

    bin_edges = np.linspace(0, rmax, bins+1, dtype=np.float64)
    cdef double [:] bin_edges_view = bin_edges

    cdef double dr = bin_edges[1] - bin_edges[0]

    DeltaV = (4. / 3.) * np.pi * (bin_edges[1:]**3. - bin_edges[:-1] ** 3.)
    cdef double [:] DeltaV_view = DeltaV

    gr = np.zeros(bins, dtype = np.float64)
    cdef double [:] gr_view = gr

    cdef double d, dd, ddiv
    cdef Py_ssize_t ii, i, j, p, k, q

    l = np.array([box[0,1] - box[0,0], box[1,1] - box[1,0], box[2,1] - box[2,0]])
    cdef double [:] l_view = l

    cdef double V = l[0] * l[1] * l[2]
    cdef double rho = (n - 1.0) / V

    with nogil, parallel():
        r_count_local_buf = <double *> malloc(sizeof(double) * bins)
        if r_count_local_buf is NULL:
            abort()

        for ii in range(bins):
            r_count_local_buf[ii] = 0.0

        for i in prange(n-1, schedule='guided'):
            for j in range(i+1, n):
                dd = 0.0
                for p in range(ndim):
                    d = positions[i, p] - positions[j, p]
                    d = d - nearbyint(d / l_view[p]) * l_view[p]
                    dd = dd + d * d
                d = sqrt(dd)
                ddiv = d / dr
                #k = int(d / dr)
                q = int(ddiv)
                if q < bins:
                    r_count_local_buf[q] += 2.0

        with gil:
            for k in range(bins):
                r_count_view[k] += r_count_local_buf[k]

        free(r_count_local_buf)

    for k in range(bins):
        gr_view[k] = r_count_view[k] / (DeltaV_view[k] * n * rho)

    return gr

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def rdf_pair_intermolecular(double [:,:] positions1 not None, double [:,:] positions2 not None, int [:] mol_index1 not None, int [:] mol_index2 not None, double rmax, int bins, double [:,:] box not None, double V):
    cdef Py_ssize_t n1 = positions1.shape[0]
    cdef Py_ssize_t n2 = positions2.shape[0]

    cdef Py_ssize_t ndim = positions1.shape[1]

    r_count = np.zeros(bins, dtype=np.float64)
    cdef double [:] r_count_view = r_count

    bin_edges = np.linspace(0, rmax, bins+1, dtype=np.float64)
    cdef double [:] bin_edges_view = bin_edges

    cdef double dr = bin_edges[1] - bin_edges[0]

    DeltaV = (4. / 3.) * np.pi * (bin_edges[1:]**3. - bin_edges[:-1] ** 3.)
    cdef double [:] DeltaV_view = DeltaV

    gr = np.zeros(bins, dtype = np.float64)
    cdef double [:] gr_view = gr

    cdef double d, dd, ddiv
    cdef Py_ssize_t ii, i, j, p, k, q

    l = np.array([box[0,1] - box[0,0], box[1,1] - box[1,0], box[2,1] - box[2,0]])
    cdef double [:] l_view = l

    #cdef double V = l[0] * l[1] * l[2]

    with nogil, parallel():
        r_count_local_buf = <double *> malloc(sizeof(double) * bins)
        if r_count_local_buf is NULL:
            abort()

        for ii in range(bins):
            r_count_local_buf[ii] = 0.0

        for i in prange(n1, schedule='guided'):
            for j in range(n2):
                dd = 0.0
                if mol_index1[i] == mol_index2[j]:
                    continue
                for p in range(ndim):
                    d = positions1[i, p] - positions2[j, p]
                    d = d - nearbyint(d / l_view[p]) * l_view[p]
                    dd = dd + d * d
                d = sqrt(dd)
                ddiv = d / dr
                q = int(ddiv)
                if q < bins:
                    r_count_local_buf[q] += 2.0

        with gil:
            for k in range(bins):
                r_count_view[k] += r_count_local_buf[k]

        free(r_count_local_buf)

    for k in range(bins):
        gr_view[k] = V * r_count_view[k] / (DeltaV_view[k] * n1 * n2)

    return gr

