import cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
DTYPE2 = np.int32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def contactmapevolution(np.ndarray[DTYPE_t, ndim=2] frame1, np.ndarray[DTYPE_t, ndim=2] frame2, double cutoff):
    cdef DTYPE_t c1,c2,change,ratio
    c1 = 0.0
    c2 = 0.0
    change = 0.0
    ratio = 0.0
    cdef int N1 = frame1.shape[0]
    cdef int dim1 = frame1.shape[1]

    cdef int N2 = frame2.shape[0]
    cdef int dim2 = frame2.shape[1]

    assert N1 == N2
    assert dim1 == dim2

    cdef DTYPE_t tmp1, tmp2, dsquare1, dsquare2
    cdef int i, j, k
    for i in xrange(N1-1):
        for j in xrange(i+1, N1):
             dsquare1 = 0.0
             dsquare2 = 0.0
             for k in range(dim1):
                 tmp1 = frame1[i,k] - frame1[j,k]
                 tmp2 = frame2[i,k] - frame2[j,k]
                 dsquare1 += tmp1 * tmp1
                 dsquare2 += tmp2 * tmp2
             if dsquare1 <= cutoff * cutoff and dsquare2 > cutoff * cutoff:
                 c1 += 1.0
                 change += 1.0
             elif dsquare1 > cutoff * cutoff and dsquare2 <= cutoff * cutoff:
                 c2 += 1.0
                 change += 1.0
             elif dsquare1 <= cutoff * cutoff and dsquare2 <= cutoff * cutoff:
                 c1 += 1.0
                 c2 += 1.0

    ratio = change/(c1+c2)

    return ratio
