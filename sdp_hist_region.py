import numpy as np
try:
	from .core import _sdp_hist_region as _sdp_hist_region
except ImportError:
	from core import _sdp_hist_region as _sdp_hist_region

# define 0 function
def sdphistregion0(frame_t, radius):
	if frame_t.dtype == np.float64:
		frame_t = frame_t.astype(np.float32, copy=False)
	radius = float(radius)
	return _sdp_hist_region.sdp_hist_square_region(frame_t, radius)

def sdphistregion(radius):
	return lambda frame_t: sdphistregion0(frame_t, radius)
