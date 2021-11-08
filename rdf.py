# rdf.py
# compute the radial distribution function
# rdf(i) = h_{ab}(i)/\Delta V(i) \rho_b N_a
# when a == b, \rho_b = (N_b - 1)/V N_a = N_b = N
# h_{ab}(i) does not contain self pair distances, i.e. only non-zero values
# h_{ab}(i) does not duplicates, that is both r_{ij} and r_{ji} is counted

import numpy as np
import scipy.spatial
from .core import _pdist_pbc

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
