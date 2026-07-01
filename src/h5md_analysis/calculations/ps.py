import numpy as np
from .._core import _ps as _ps

# define ps0 function
def ps0(frame_t, cutoff):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _ps.ps(frame_t, cutoff)

# define ps lambda function
def ps(cutoff):
	return lambda frame_t: ps0(frame_t, cutoff)
