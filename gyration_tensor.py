import numpy as np
from core import _gyration_tensor as _gt

# define loop gyration tensor function
def gyration_tensor(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _gt.compute_gyration_tensor(frame_t)
