import numpy as np
try:
    from .core import _loop_gyration_tensor as _lgt
except ImportError:
    from core import _loop_gyration_tensor as _lgt

# define loop gyration tensor function
def loop_gyration_tensor(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _lgt.compute_loop_gyration_tensor(frame_t)
