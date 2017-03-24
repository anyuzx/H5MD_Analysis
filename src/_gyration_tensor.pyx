import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
DTYPE2 = np.int32
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_gyration_tensor(np.ndarray[DTYPE_t, ndim=2] frame):
  cdef int N = frame.shape[0]
  cdef np.ndarray[DTYPE2_t, ndim = 1] slist = np.unique(np.int32(np.exp(np.linspace(np.log(2),np.log(N), 100))))
  cdef int slist_size = slist.shape[0]

  cdef np.ndarray[DTYPE_t, ndim = 2] gyration_tensor = np.zeros((slist_size, 6), dtype=DTYPE)
  cdef int index, s, i, k
  cdef DTYPE_t xmean, ymean, zmean, rxx, ryy, rzz, rxy, rxz, ryz
  index = 0
  for s in slist:
    for i in xrange(N-s+1):
      xmean = np.mean(frame[i:i+s], axis=0)[0]
      ymean = np.mean(frame[i:i+s], axis=0)[1]
      zmean = np.mean(frame[i:i+s], axis=0)[2]
      rxx, ryy, rzz, rxy, rxz, ryz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      for k in xrange(s):
        rxx += (1.0/s) * (frame[i+k, 0] - xmean) * (frame[i+k, 0] - xmean)
        ryy += (1.0/s) * (frame[i+k, 1] - ymean) * (frame[i+k, 1] - ymean)
        rzz += (1.0/s) * (frame[i+k, 2] - zmean) * (frame[i+k, 2] - zmean)
        rxy += (1.0/s) * (frame[i+k, 0] - xmean) * (frame[i+k, 1] - ymean)
        rxz += (1.0/s) * (frame[i+k, 0] - xmean) * (frame[i+k, 2] - zmean)
        ryz += (1.0/s) * (frame[i+k, 1] - ymean) * (frame[i+k, 2] - zmean)

      gyration_tensor[index, 0] += rxx/(N-s+1)
      gyration_tensor[index, 1] += ryy/(N-s+1)
      gyration_tensor[index, 2] += rzz/(N-s+1)
      gyration_tensor[index, 3] += rxy/(N-s+1)
      gyration_tensor[index, 4] += rxz/(N-s+1)
      gyration_tensor[index, 5] += ryz/(N-s+1)

    index += 1

  return gyration_tensor
