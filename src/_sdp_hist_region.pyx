"""
_sdp_hist_region.pyx

Similar to the _sdp_hist.pyx.
The difference is that this module
only compute the distance between beads
in a region if the whole segment between
them are within certain distance from the
COM(center of mass).
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
def sdp_hist_square_region(np.ndarray[DTYPE_t, ndim=2] frame, double radius):
	cdef int N = frame.shape[0]
	cdef int dim = frame.shape[1]
	cdef i, j, l, k
	cdef dsquare, tmp

	# compute the center of mass
	cdef np.ndarray[DTYPE_t, ndim=2] com = np.mean(frame_t, axis=0)
	# compute the distance between each beads to the center of mass
	cdef np.ndarray[DTYPE_t, ndim=1] radial_dist = np.sqrt(np.sum(np.power(frame - com, 2.0), axis=1))

	DISTANCE_LST = [[] for i in range(N-1)]

	cdef int start = 0

	while start <= N - 2:
		for i in xrange(start + 1, N):
			if radial_dist[i] <= radius:
				for l in xrange(start, i):
					dsquare = 0.0
					for k in range(dim):
						tmp = frame[l, k] - frame[i, k]
						dsquare += tmp * tmp
					DISTANCE_LST[i - l - 1].append(dsquare)
			else:
				for j in xrange(i + 1, N):
					if radial_dist[j] <= radius:
						start = j
						break

	return DISTANCE_LST