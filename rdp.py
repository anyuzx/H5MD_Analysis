# rdp.py
# Radial Density Profile
# This module calculate the radial density profile based on atom index
# can be used to calculate the radial density profile for multiple
# types of atoms

import numpy as np

def rdp0(frame_t, index, rmax, dr):
    if type(index) != list:
        raise ValueError('index is not a list\n')

    # get number of atoms in frame
    natoms = frame_t.shape[0]
    # get center of mass
    com_t = np.mean(frame_t, axis=0)
    # calculate the distance of each atom to center of mass
    radial_dist = np.sqrt(np.sum(np.power(frame_t - com_t, 2.0), axis=1))
    # get bins_edge
    bins_center = np.linspace(0.0, rmax, int(rmax/dr))
    bins_center = np.diff(bins_center)/2.0 + bins_center[:-1]
    actual_dr = bins_center[1] - bins_center[0]

    density = []
    for atom_index in index:
        atom_index = np.array(atom_index, dtype=np.int) - 1
        hist = np.histogram(radial_dist[atom_index], bins=np.linspace(0.0, rmax, int(rmax/dr)))
        density_temp = hist/(4*np.pi*(bins_center**2)*(atom_index.shape[0]/((4.0/3.0)*np.pi*rmax**3.0))*actual_dr)
        density.append(np.column_stack((bins_center, density_temp)))

    return np.array(density)

def rdp(index, rmax, dr=0.1):
    return lambda frame_t: rdp0(frame_t, index, rmax, dr)
