# this function compute the structural factor in fourier space (k-space)
# this function use Lebedev quadrature https://en.wikipedia.org/wiki/Lebedev_quadrature
import numpy as np
from .core import _sk_direct

def sk0(frame_t, index1, index2, kmax, box, region=None):
    # currently region can only used to define the region along the z-dimention
    # region should be a list, like [a,b] where a and b define the lower and upper boundary of the region, respectively
    atom_index1 = np.array(index1, dtype=int)
    atom_index2 = np.array(index2, dtype=int)
    natoms1 = atom_index1.shape[0]
    natoms2 = atom_index2.shape[0]

    if box is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
        l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    
    if region is not None:
        l[2] = region[1] - region[0]

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False
    else:
        print('atom index 1 and atom index 2 can only be either all the same or all different')
        return

    if isSelf:
        frame_t_select = frame_t[atom_index1]
        if region is not None:
            frame_t_select = frame_t_select[(frame_t_select[:,2] >= region[0]) & (frame_t_select[:,2] <= region[1])]
        sk_res = _sk_direct.sk_direct_self(frame_t_select, kmax, l)
    else:
        frame_t_select1 = frame_t[atom_index1]
        frame_t_select2 = frame_t[atom_index2]
        if region is not None:
            frame_t_select1 = frame_t_select1[(frame_t_select1[:,2] >= region[0]) & (frame_t_select1[:,2] <= region[1])]
            frame_t_select2 = frame_t_select2[(frame_t_select2[:,2] >= region[0]) & (frame_t_select2[:,2] <= region[1])]
        sk_res = _sk_direct.sk_direct_cross(frame_t_select1, frame_t_select2, kmax, l)

    return sk_res[sk_res[:,0].argsort()]

def sk_direct(index1, index2, kmax, box, region):
    return lambda frame_t: sk0(frame_t, index1, index2, kmax, box, region)
