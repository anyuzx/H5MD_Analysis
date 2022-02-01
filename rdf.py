# rdf.py
# compute the radial distribution function
# rdf(i) = h_{ab}(i)/\Delta V(i) \rho_b N_a
# when a == b, \rho_b = (N_b - 1)/V N_a = N_b = N
# h_{ab}(i) does not contain self pair distances, i.e. only non-zero values
# h_{ab}(i) does not duplicates, that is both r_{ij} and r_{ji} is counted

import numpy as np
import scipy.spatial
from .core import _pdist_pbc
from .core import _rdf

def rdf0(frame_t, index1, index2, rmax, bins, V, pbc, box):
    if V is None and box is None:
        print('Please specify either volume or box dimension')
        exit(0)

    if V is not None and box is not None:
        print('Please specify only one argument: volume or box dimension')
        exit(0)
    # get number of atoms in frame
    atom_index1 = np.array(index1, dtype=int) - 1
    atom_index2 = np.array(index2, dtype=int) - 1
    natom1 = len(atom_index1)
    natom2 = len(atom_index2)

    if np.all(atom_index1 == atom_index2):
        isSelf = True
    elif np.all(atom_index1 != atom_index2):
        isSelf = False
    else:
        print('atom index 1 and atom index 2 can only be either all the same or all different')
        return

    if box is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
        l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
        V = l[0] * l[1] * l[2]

    if pbc is False:
        distances = (scipy.spatial.distance.cdist(frame_t[atom_index1], frame_t[atom_index2])).flatten()
        distances = distances[np.nonzero(distances)]
    else:
        if box is None:
            print('Please specify the box dimension when periodic boundary condition is used')
            exit(0)
        frame_t_shift = frame_t - np.array([xlo, ylo, zlo])
        distances = _pdist_pbc.cdist_cy_omp(frame_t_shift[atom_index1], frame_t_shift[atom_index2], l)
        distances = distances[np.nonzero(distances)]
    
    if isSelf:
        rho_2 = (natom2 - 1) / V
    else:
        rho_2 = natom2 / V

    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, rmax))
    #DeltaV = 4. * np.pi * (bin_edges[1] - bin_edges[0]) * bin_edges[1:] ** 2.
    DeltaV  = (4. / 3.) * np.pi * (bin_edges[1:]**3. - bin_edges[:-1] ** 3.)

    density = hist/(DeltaV*(natom1*rho_2))

    return np.array(np.column_stack((bin_edges[1:],density)))

def rdf(index1, index2, rmax, bins, V=None, pbc=False, box=None):
    return lambda frame_t: rdf0(frame_t, index1, index2, rmax, bins, V, pbc, box)

def rdf_cython0(frame_t, index, rmax, bins, box):
    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]

    frame_t_shift = frame_t - np.array([xlo, ylo, zlo])

    bin_edges = np.linspace(0, rmax, bins+1)

    gr = _rdf.rdf_self(frame_t_shift[index], rmax, int(bins), box)
    return np.column_stack((bin_edges[1:], gr))

def rdf_cython(index, rmax, bins, box):
    return lambda frame_t: rdf_cython0(frame_t, index, rmax, bins, box)

def rdf_intermolecular_cython0(frame_t, index1, index2, mol_index1, mol_index2, rmax, bins, box, region=None):
    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]

    frame_t_select1 = frame_t[index1]
    frame_t_select2 = frame_t[index2]

    mol_index1_select = np.copy(mol_index1)
    mol_index2_select = np.copy(mol_index2)

    if region is not None:
        V = (yhi - ylo) * (xhi - xlo) * (region[1] - region[0])
    else:
        V = (yhi - ylo) * (xhi - xlo) * (zhi - zlo)


    if region is not None:
        select_boolean_1 = (frame_t_select1[:,2] >= region[0]) & (frame_t_select1[:,2] <= region[1])
        select_boolean_2 = (frame_t_select2[:,2] >= region[0]) & (frame_t_select2[:,2] <= region[1])

        frame_t_select1 = frame_t_select1[select_boolean_1]
        frame_t_select2 = frame_t_select2[select_boolean_2]

        mol_index1_select = mol_index1[select_boolean_1]
        mol_index2_select = mol_index2[select_boolean_2]

    # shift the coordinates for periodic boundary computation
    frame_t_select1 = frame_t_select1 - np.array([xlo, ylo, zlo])
    frame_t_select2 = frame_t_select2 - np.array([xlo, ylo, zlo])

    bin_edges = np.linspace(0, rmax, bins+1)

    gr = _rdf.rdf_pair_intermolecular(frame_t_select1, frame_t_select2, mol_index1_select, mol_index2_select, rmax, int(bins), box, V)
    return np.column_stack((bin_edges[1:], gr))

def rdf_intermolecular_cython(index1, index2, mol_index1, mol_index2, rmax, bins, box, region):
    return lambda frame_t: rdf_intermolecular_cython0(frame_t, index1, index2, mol_index1, mol_index2, rmax, bins, box, region)
