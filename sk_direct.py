# this function compute the structural factor in fourier space (k-space)
# this function use Lebedev quadrature https://en.wikipedia.org/wiki/Lebedev_quadrature
import numpy as np
from .core import _sk_direct

def sk0(frame_t, index1, index2, kmax, box):
    atom_index1 = np.array(index1, dtype=int)
    atom_index2 = np.array(index2, dtype=int)
    natoms1 = atom_index1.shape[0]
    natoms2 = atom_index2.shape[0]

    if box is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
        l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False
    else:
        print('atom index 1 and atom index 2 can only be either all the same or all different')
        return

    frame_t_shift = frame_t - np.array([xlo, ylo, zlo])

    if isSelf:
        sk_res = _sk_direct.sk_direct_self(frame_t_shift[atom_index1], kmax, l)
    else:
        sk_res = _sk_direct.sk_direct_cross(frame_t_shift[atom_index1], frame_t_shift[atom_index2], kmax, l)

    return sk_res[sk_res[:,0].argsort()]

def sk_direct(index1, index2, kmax, box):
    return lambda frame_t: sk0(frame_t, index1, index2, kmax, box)
