import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef np.ndarray[DTYPE_t, ndim=2] loop_list = \
np.array([[43,396],[43,582],[143,396],[948,1110],[1407,1570],
[1628,2120],[2355,2562],[2409,2562],[2622,2917],[3059,3106],
[3307,3378],[3307,3630],[3307,3471],[4131,4175],[4131,4307],
[4445,5012],[4445,4710],[5058,5548],[6318,6766],[6318,6408],
[6318,6595],[6408,6595],[6647,6766],[7605,8907],[7917,8644],
[7917,8743],[7917,8907],[8743,8907],[8921,9008],[9157,9396],
[9481,9562],[9510,9562]])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_gyration_tensor(np.ndarray[DTYPE_t, ndim=2] frame):
  cdef int N = frame.shape[0]
  cdef int dim = frame.shape[1]

  cdef int n_loop = loop_list.shape[0]

  cdef np.ndarray[DTYPE_t, ndim = 2] gyration_tensor = np.zeros((n_loop,3,3), dtype=DTYPE)
  cdef DTYPE_t tmp, dsquare
  cdef int i, k
  cdef int loop_start, loop_end, loop_size
  cdef DTYPE_t xmean, ymean, zmean
  for i in xrange(n_loop):
    loop_start = loop_list[i,0] - 1
    loop_end = loop_list[i,1] - 1
    loop_size = loop_list[i,1] - loop_list[i,0] + 1
    xmean = np.mean(frame, axis=0)[0]
    ymean = np.mean(frame, axis=0)[1]
    zmean = np.mean(frame, axis=0)[2]
    for k in xrange(loop_size - 1):
      gyration_tensor[i,0,0] += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,0] - xmean)
      gyration_tensor[i,1,1] += (1.0/loop_size) * (frame[k+loop_start,1] - ymean) * (frame[k+loop_start,1] - ymean)
      gyration_tensor[i,2,2] += (1.0/loop_size) * (frame[k+loop_start,2] - zmean) * (frame[k+loop_start,2] - zmean)
      gyration_tensor[i,0,1] += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,1] - ymean)
      gyration_tensor[i,0,2] += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,2] - zmean)
      gyration_tensor[i,1,2] += (1.0/loop_size) * (frame[k+loop_start,1] - ymean) * (frame[k+loop_start,2] - zmean)

      gyration_tensor[i,0,1] = gyration_tensor[i,1,0]
      gyration_tensor[i,0,2] = gyration_tensor[i,2,0]
      gyration_tensor[i,1,2] = gyration_tensor[i,2,1]

  return gyration_tensor
