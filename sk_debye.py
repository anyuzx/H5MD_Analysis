# function to compute static structural factor S(k)
# This function use Debye scattering formula

# S_{aa}(k) = 1 + (1/N_a)*\sum_{i!=j}^{N_a}Sin(k r_ij) / (k r_ij)
# S_{ab}(k) = (1/(N_a + N_b)) * \sum_{i=1}^{N_a}\sum_{j=1}^{N_b}Sin(k r_ij) / (k r_ij)
# this module is able to compute S(k) for non-periodic or periodic boundary condition
import numpy as np
from .core import _pdist_pbc
from .core import _sk_debye

def sk0_debye(frame_t, index1, index2, kmax, dk, box, kmin=0.0):
    # get number of atoms

    atom_index1 = np.array(index1, dtype=int)
    atom_index2 = np.array(index2, dtype=int)
    natoms1 = atom_index1.shape[0]
    natoms2 = atom_index2.shape[0]

    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
    l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

    k = np.arange(kmin, kmax, dk)

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False
    else:
        print('atom index 1 and atom index 2 can only be either all the same or all different')
        return

    frame_t_shift = frame_t - np.array([xlo, ylo, zlo])

    if isSelf:
        sk_res = _sk_debye.sk_debye_self(frame_t_shift[atom_index1], k, l)
    else:
        sk_res = _sk_debye.sk_debye_cross(frame_t_shift[atom_index1], frame_t_shift[atom_index2], k, l)

    return np.column_stack((k, sk_res))

def sk_debye(index1, index2, kmax, dk, box):
    return lambda frame_t: sk0_debye(frame_t, index1, index2, kmax, dk, box)

def sk0_histogram_method(frame_t, index1, index2, kmax, dk, box, kmin, bins):
    # get number of atoms

    atom_index1 = np.array(index1, dtype=int)
    atom_index2 = np.array(index2, dtype=int)
    natoms1 = atom_index1.shape[0]
    natoms2 = atom_index2.shape[0]

    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
    l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])

    k = np.arange(kmin, kmax, dk)

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False
    else:
        print('atom index 1 and atom index 2 can only be either all the same or all different')
        return

    frame_t_shift = frame_t - np.array([xlo, ylo, zlo])

    if isSelf:
        pdistances = _pdist_pbc.pdist_cy_omp(frame_t_shift[atom_index1], l)
    else:
        pdistances = _pdist_pbc.cdist_cy_omp(frame_t_shift[atom_index1], frame_t_shift[atom_index2], l)

    hist, bin_edges = np.histogram(pdistances, bins=bins, density=False)
    bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2.

    if isSelf:
        sk_res = 1 + (2. / natoms1) * np.sum(hist * np.sinc(k[:, np.newaxis] * bin_center / np.pi), axis=1)
    else:
        sk_res = (1.0 / (natoms1 + natoms2)) * np.sum(hist * np.sinc(k[:, np.newaxis] * bin_center / np.pi), axis=1)

    return np.column_stack((k, sk_res))

def sk_histogram_method(index1, index2, kmax, dk, box, kmin=0.0, bins=100):
    return lambda frame_t: sk0_histogram_method(frame_t, index1, index2, kmax, dk, box, kmin, bins)
