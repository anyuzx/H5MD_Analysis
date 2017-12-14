import numpy as np
from core import _sdp_hist as _sdp_hist


# define sdphist0 function
def sdphistsquare0(frame_t, slist):
    if frame_t.dtype == 'float64':
        frame_t = np.float32(frame_t)
    slist = np.int32(slist)

    return _sdp_hist.sdp_hist_square(frame_t, slist)


# define sdp hist lambda function, this is the actual function passed to class LammpsH5MD routine
def sdphistsquare(slist):
    return lambda frame_t: sdphistsquare0(frame_t, slist)
