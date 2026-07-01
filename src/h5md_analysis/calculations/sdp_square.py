import numpy as np
from .._core import _sdp as _sdp

# define subchain distance profile
def sdp_square(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _sdp.sdpsquare(frame_t)
