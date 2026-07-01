import numpy as np
try:
    from .core import _contactmap as _cmap
except ImportError:
    from core import _contactmap as _cmap


# define contactmap0 function
def contactmap0(frame_t, cutoff):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _cmap.contactmap(frame_t, cutoff)


# define contactmap lambda function, this is the actual function passed to class LammpsH5MD routine
def contactmap(cutoff):
    return lambda frame_t: contactmap0(frame_t, cutoff)
