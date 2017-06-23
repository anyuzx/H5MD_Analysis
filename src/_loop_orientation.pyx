import numpy as np
import cython
cimport numpy as np

"""
This module compute the orientation of each loop domains
The shape of loop domains is prolate. Thus computing gyration
tensor to get the eigenvalues. The largest eigenvalue corresponds
to the long axis. The eigenvector corresponding to the largest
eigenvalue represents the vector of long axis (only the orientation
information is used). This module will ouput the x, y, z elements of
the eigenvector of all thirty-two loop domains
"""

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
def compute_loop_orientation(np.ndarray[DTYPE_t, ndim=2] frame):
  cdef int N = frame.shape[0]
  cdef int dim = frame.shape[1]

  cdef int n_loop = 32

  cdef np.ndarray[DTYPE_t, ndim = 2] long_axis_vector_array = np.zeros((n_loop, 3), dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim = 2] rt_mtx = np.zeros((3,3), dtype=DTYPE)
  cdef np.ndarray[DTYPE_t, ndim = 2] eigenvector = np.zeros((3,3), dtype=DTYPE)

  cdef int i, k
  cdef int loop_start, loop_end, loop_size
  cdef DTYPE_t xmean, ymean, zmean, rxx, ryy, rzz, rxy, rxz, ryz

  for i in xrange(n_loop):
    loop_start = loop_list[i][0] - 1
    loop_end = loop_list[i][1] - 1
    loop_size = loop_list[i][1] - loop_list[i][0] + 1
    xmean = np.mean(frame[loop_start:loop_end+1], axis=0)[0]
    ymean = np.mean(frame[loop_start:loop_end+1], axis=0)[1]
    zmean = np.mean(frame[loop_start:loop_end+1], axis=0)[2]
    rxx, ryy, rzz, rxy, rxz, ryz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for k in xrange(loop_size - 1):
      rxx += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,0] - xmean)
      ryy += (1.0/loop_size) * (frame[k+loop_start,1] - ymean) * (frame[k+loop_start,1] - ymean)
      rzz += (1.0/loop_size) * (frame[k+loop_start,2] - zmean) * (frame[k+loop_start,2] - zmean)
      rxy += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,1] - ymean)
      rxz += (1.0/loop_size) * (frame[k+loop_start,0] - xmean) * (frame[k+loop_start,2] - zmean)
      ryz += (1.0/loop_size) * (frame[k+loop_start,1] - ymean) * (frame[k+loop_start,2] - zmean)

    rt_mtx[0,0] = rxx
    rt_mtx[1,1] = ryy
    rt_mtx[2,2] = rzz
    rt_mtx[0,1] = rxy
    rt_mtx[0,2] = rxz
    rt_mtx[1,0] = rxy
    rt_mtx[1,2] = ryz
    rt_mtx[2,0] = rxz
    rt_mtx[2,1] = ryz

    eigenvalue, eigenvector = np.linalg.eigh(rt_mtx)

    long_axis_vector_array[i, 0] = eigenvector[0,-1]
    long_axis_vector_array[i, 1] = eigenvector[1,-1]
    long_axis_vector_array[i, 2] = eigenvector[2,-1]

  return long_axis_vector_array
