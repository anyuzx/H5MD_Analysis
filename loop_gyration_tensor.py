import numpy as np
from core import _loop_gyration_tensor as _lgt

# define loop gyration tensor function
def gyration_tensor(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _lgt.compute_gyration_tensor(frame_t)
