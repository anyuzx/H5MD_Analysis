import numpy as np
from .._core import _type_gyration_tensor as _tgt

# define type gyration tensor function
def type_gyration_tensor(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _tgt.compute_type_gyration_tensor(frame_t)
