import numpy as np
from .._core import _distmap as _dmap


# define distmap function
def distmap(frame_t):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)

    return _dmap.distmap(frame_t)

# define distmap square function
# return dij^2 not dij
def distmap_square(frame_t):
	if frame_t.dtype == np.float64:
		frame_t = frame_t.astype(np.float32, copy=False)

	return _dmap.distmap_square(frame_t)
