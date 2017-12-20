# rdp_atom.py
# This module calculate the distance between certain atoms to the center of mass
# the argument index is used to specify atom ids you want to compute
# the module return a array contains distance to the COM

import numpy as np

def rdp_atom0(frame_t, index):
    compute_all = False
    if index == 'all':
        compute_all = True
    elif type(index) != list:
        raise ValueError("index is not list. Please either specify the index or use 'all'\n")

    # get number of atoms in frame
    natoms = frame_t.shape[0]
    # get center of mass
    com_t = np.mean(frame_t, axis=0)
    # calculate the distance of each atom to center of mass
    radial_dist = np.sqrt(np.sum(np.power(frame_t - com_t, 2.0), axis=1))

    if compute_all:
        return radial_dist
    else:
        index = np.array(index, dtype=np.int) - 1
        return radial_dist[index]

def rdp_atom(index):
    return lambda frame_t: rdp0(frame_t, index)
