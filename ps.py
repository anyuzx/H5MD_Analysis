import numpy as np
from core import _ps as _ps

# define ps0 function
def ps0(frame_t, cutoff):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _ps.ps(frame_t, cutoff)

# define ps lambda function
def ps(cutoff):
	return lambda frame_t: ps0(frame_t, cutoff)
