import numpy as np
try:
    from .core import _loop_orientation as _lo
except ImportError:
    from core import _loop_orientation as _lo

# define loop orientation function
def loop_orientation(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _lo.compute_loop_orientation(frame_t)
