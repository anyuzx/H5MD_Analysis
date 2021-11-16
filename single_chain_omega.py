# this module compute the /omega function for a single chain
import numpy as np
import scipy
import scipy.spatial
from .core import _pdist_pbc
from tqdm import tqdm

def xyz2omega(xyz1, xyz2, karray, l):
    natoms1 = xyz1.shape[0]
    natoms2 = xyz2.shape[0]
    #cdist = scipy.spatial.distance.cdist(xyz1, xyz2).flatten()
    cdist = _pdist_pbc.cdist_cy_omp(xyz1, xyz2, l)
    krij = karray[:, np.newaxis] * cdist
    krij_sum = np.sum(np.sinc(krij / np.pi), axis=1)
    return (1.0 / (natoms1 + natoms2)) * krij_sum

def single_chain_omega0(frame_t, atom_index1, atom_index2, nmol, kmax, dk, box):
    # atom_index is the index for computing in a single molecule
    # len(atom_index) == mol_length
    # k is the array for wave vector
    # nmol: number of molecules in the system
    karray = np.arange(0, kmax, dk)
    natoms = frame_t.shape[0]
    mol_length = int(natoms / nmol)

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False

    assert atom_index1.max() + 1 <= mol_length and atom_index2.max() + 1 <= mol_length, 'Atom index is not valid'

    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
    l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    
    frame_t_shift = frame_t - np.array([xlo, ylo, zlo])
    frame_t_reshaped = frame_t_shift.reshape((nmol, mol_length, 3))

    omega = np.zeros(karray.shape[0])
    for mol in frame_t_reshaped:
        # mol is array of shape (mol_length, 3)
        xyz1 = mol[atom_index1]
        xyz2 = mol[atom_index2]
        if isSelf:
            omega_single_mol = 2. * xyz2omega(xyz1, xyz2, karray, l)
        else:
            omega_single_mol = xyz2omega(xyz1, xyz2, karray, l)
        omega += omega_single_mol
    omega /= nmol

    return np.column_stack((karray, omega))

def single_chain_omega(atom_index1, atom_index2, nmol, kmax, dk, box):
    return lambda frame_t: single_chain_omega0(frame_t, atom_index1, atom_index2, nmol, kmax, dk, box)
