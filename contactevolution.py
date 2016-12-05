import numpy as np
from core import _contactevolution as _cmapevolution

def contactevolution0(frame_t1, frame_t2, cutoff):
    if frame_t1.dtype == 'float64':
        frame_t1 = np.float32(frame_t1)
    if frame_t2.dtype == 'float64':
        frame_t2 = np.float32(frame_t2)

    return _cmapevolution.contactmapevolution(frame_t1, frame_t2, cutoff)

def contactevolution(cutoff):
    return lambda frame_t1, frame_t2: contactevolution0(frame_t1, frame_t2, cutoff)