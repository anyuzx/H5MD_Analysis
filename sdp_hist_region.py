import numpy as np
from core import _sdp_hist_region as _sdp_hist_region

# define 0 function
def sdphistregion0(frame_t, radius):
	if frame_t.dtype == 'float64':
		frame_t = np.float32(frame_t)
	radius = float(radius)
	return _sdp_hist_region.sdp_hist_square_region(frame_t, radius)

def sdphistregion(radius):
	return lambda frame_t: sdphistregion0(frame_t, radius)