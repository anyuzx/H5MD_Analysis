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

  cdef np.ndarray[DTYPE_t, ndim = 2] shape_array = np.zeros((slist_size, 3), dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim = 2] rt_mtx = np.zeros((3,3), dtype=DTYPE)
  cdef int index, s, i, k
  cdef DTYPE_t xmean, ymean, zmean, rxx, ryy, rzz, rxy, rxz, ryz
  cdef DTYPE_t rgsquare, ksquare, shape
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

      rt_mtx[0,0] = rxx
      rt_mtx[1,1] = ryy
      rt_mtx[2,2] = rzz
      rt_mtx[0,1] = rxy
      rt_mtx[0,2] = rxz
      rt_mtx[1,0] = rxy
      rt_mtx[1,2] = ryz
      rt_mtx[2,0] = rxz
      rt_mtx[2,1] = ryz

      eigenvalue = np.linalg.eigvalsh(rt_mtx)
      rgsquare = np.sum(eigenvalue)
      ksquare = 1.5*(np.sum(np.power(eigenvalue,2.0)))/(np.sum(eigenvalue))**2.0 - 0.5
      shape = np.prod((eigenvalue - np.mean(eigenvalue))/np.mean(eigenvalue))

      shape_array[index, 0] += rgsquare/(N-s+1)
      shape_array[index, 1] += ksquare/(N-s+1)
      shape_array[index, 2] += shape/(N-s+1)

    index += 1

  return shape_array
