import numpy as np
from core import _type_gyration_tensor as _tgt

# define type gyration tensor function
def type_gyration_tensor(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _tgt.compute_type_gyration_tensor(frame_t)
