import numpy as np
from core import _loop_orientation as _lo

# define loop orientation function
def loop_orientation(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _lgt.compute_loop_orientation(frame_t)
