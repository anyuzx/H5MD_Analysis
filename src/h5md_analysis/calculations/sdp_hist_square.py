import numpy as np
from .._core import _sdp_hist as _sdp_hist


# define sdphist0 function
def sdphistsquare0(frame_t, slist):
    if frame_t.dtype == np.float64:
        frame_t = frame_t.astype(np.float32, copy=False)
    slist = np.int32(slist)

    return _sdp_hist.sdp_hist_square(frame_t, slist)


# define sdp hist lambda function, this is the actual function passed to class LammpsH5MD routine
def sdphistsquare(slist):
    return lambda frame_t: sdphistsquare0(frame_t, slist)
