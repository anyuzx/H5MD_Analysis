import numpy as np
from core import _distmap as _dmap


# define distmap function
def distmap(frame_t):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)

    return _dmap.distmap(frame_t)

# define distmap square function
# return dij^2 not dij
def distmap_square(frame_t):
	if frame_t.dtype == 'float64':
		frame_t = np.float32(frame_t)

	return _dmap.distmap_square(frame_t)
