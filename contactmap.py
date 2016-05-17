import numpy as np
from core import _contactmap as _cmap


# define contactmap0 function
def contactmap0(frame_t, cutoff):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _cmap.contactmap(frame_t, cutoff)


# define contactmap lambda function, this is the actual function passed to class LammpsH5MD routine
def contactmap(cutoff):
    return lambda frame_t: contactmap0(frame_t, cutoff)
