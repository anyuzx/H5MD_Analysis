import numpy as np
from core import _sdp as _sdp

# define subchain distance profile
def sdp(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _sdp.sdp(frame_t)
