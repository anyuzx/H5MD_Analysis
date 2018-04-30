# rdf.py
# compute the radial distribution function

import numpy as np
import scipy.spatial

def rdf0(frame_t, index1, index2, rmax, dr, V):
    # get number of atoms in frame

    atom_index1 = np.array(index1, dtype=np.int) - 1
    atom_index2 = np.array(index2, dtype=np.int) - 1
    natom1 = len(atom_index1)
    natom2 = len(atom_index2)

    distances = (scipy.spatial.distance.cdist(frame_t[atom_index1], frame_t[atom_index2])).flatten()
    distances = distances[np.nonzero(distances)]

    # get bins_edge
    bins_center = np.linspace(0.0, rmax, int(rmax/dr))
    bins_center = np.diff(bins_center)/2.0 + bins_center[:-1]
    actual_dr = bins_center[1] - bins_center[0]

    hist, bin_edges = np.histogram(distances, bins=np.linspace(0.0, rmax, int(rmax/dr)))
    density = (hist*V)/(4.0*np.pi*(bins_center**2.0)*(natom1*natom2)*actual_dr)

    return np.array(np.column_stack((bins_center,density)))

def rdf(index1, index2, rmax, dr, V):
    return lambda frame_t: rdp0(frame_t, index1, index2, rmax, dr, V)
