import numpy as np
import cython
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef int loop_list[32][2]
loop_list[0][:] = [43,396]
loop_list[1][:] = [43,582]
loop_list[2][:] = [143,396]
loop_list[3][:] = [948,1110]
loop_list[4][:] = [1407,1570]
loop_list[5][:] = [1628,2120]
loop_list[6][:] = [2355,2562]
loop_list[7][:] = [2409,2562]
loop_list[8][:] = [2622,2917]
loop_list[9][:] = [3059,3106]
loop_list[10][:] = [3307,3378]
loop_list[11][:] = [3307,3630]
loop_list[12][:] = [3307,3471]
loop_list[13][:] = [4131,4175]
loop_list[14][:] = [4131,4307]
loop_list[15][:] = [4445,5012]
loop_list[16][:] = [4445,4710]
loop_list[17][:] = [5058,5548]
loop_list[18][:] = [6318,6766]
loop_list[19][:] = [6318,6408]
loop_list[20][:] = [6318,6595]
loop_list[21][:] = [6408,6595]
loop_list[22][:] = [6647,6766]
loop_list[23][:] = [7605,8907]
loop_list[24][:] = [7917,8644]
loop_list[25][:] = [7917,8743]
loop_list[26][:] = [7917,8907]
loop_list[27][:] = [8743,8907]
loop_list[28][:] = [8921,9008]
loop_list[29][:] = [9157,9396]
loop_list[30][:] = [9481,9562]
loop_list[31][:] = [9510,9562]

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_gyration_tensor(np.ndarray[DTYPE_t, ndim=2] frame):
  cdef int N = frame.shape[0]
  cdef int dim = frame.shape[1]

  cdef int n_loop = 32

  cdef np.ndarray[DTYPE_t, ndim = 3] gyration_tensor = np.zeros((n_loop, 3, 3), dtype=DTYPE)
  cdef int i, k
  cdef int loop_start, loop_end, loop_size
  cdef DTYPE_t xmean, ymean, zmean
  for i in xrange(n_loop):
    loop_start = loop_list[i][0] - 1
    loop_end = loop_list[i][1] - 1
    loop_size = loop_list[i][1] - loop_list[i][0] + 1
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
