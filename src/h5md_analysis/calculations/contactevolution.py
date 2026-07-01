import numpy as np
from .._core import _contactevolution as _cmapevolution

def contactevolution0(frame_t1, frame_t2, cutoff):
    if frame_t1.dtype == np.float64:
        frame_t1 = frame_t1.astype(np.float32, copy=False)
    if frame_t2.dtype == np.float64:
        frame_t2 = frame_t2.astype(np.float32, copy=False)

    return _cmapevolution.contactmapevolution(frame_t1, frame_t2, cutoff)

def contactevolution(cutoff):
    return lambda frame_t1, frame_t2: contactevolution0(frame_t1, frame_t2, cutoff)
