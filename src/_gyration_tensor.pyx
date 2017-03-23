import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_gyration_tensor(np.ndarray[DTYPE_t, ndim=2] frame):
  cdef int N = frame.shape[0]

  cdef np.ndarray[DTYPE_t, ndim = 2] gyration_tensor = np.zeros((N-1, 6), dtype=DTYPE)
  cdef int i, j, k, size
  cdef DTYPE_t xmean, ymean, zmean, rxx, ryy, rzz, rxy, rxz, ryz
  for i in xrange(N-1):
    for j in xrange(i+1, N):
      size = j-i+1
      xmean = np.mean(frame[i:j+1], axis=0)[0]
      ymean = np.mean(frame[i:j+1], axis=0)[1]
      zmean = np.mean(frame[i:j+1], axis=0)[2]
      rxx, ryy, rzz, rxy, rxz, ryz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      for k in xrange(size-1):
        rxx += (1.0/size) * (frame[k+i, 0] - xmean) * (frame[k+i, 0] - xmean)
        ryy += (1.0/size) * (frame[k+i, 1] - ymean) * (frame[k+i, 1] - ymean)
        rzz += (1.0/size) * (frame[k+i, 2] - zmean) * (frame[k+i, 2] - zmean)
        rxy += (1.0/size) * (frame[k+i, 0] - xmean) * (frame[k+i, 1] - ymean)
        rxz += (1.0/size) * (frame[k+i, 0] - xmean) * (frame[k+i, 2] - zmean)
        ryz += (1.0/size) * (frame[k+i, 1] - ymean) * (frame[k+i, 2] - zmean)

      gyration_tensor[j-i-1, 0] += rxx/(N-(j-i))
      gyration_tensor[j-i-1, 1] += ryy/(N-(j-i))
      gyration_tensor[j-i-1, 2] += rzz/(N-(j-i))
      gyration_tensor[j-i-1, 3] += rxy/(N-(j-i))
      gyration_tensor[j-i-1, 4] += rxz/(N-(j-i))
      gyration_tensor[j-i-1, 5] += ryz/(N-(j-i))

  return gyration_tensor
