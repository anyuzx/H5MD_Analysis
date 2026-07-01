import numpy as np
from .._core import _gyration_tensor as _gt

# define loop gyration tensor function
def gyration_tensor(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _gt.compute_gyration_tensor(frame_t)
