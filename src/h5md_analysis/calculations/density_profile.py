import numpy as np

def density_profile0(frame_t, bins, box, along_dimension):
    dimension_map = {'x':0,  'y':1, 'z': 2}
    single_coords = frame_t[:, dimension_map[along_dimension]]

    xlo, xhi, ylo, yhi, zlo, zhi = box[0,0], box[0,1], box[1,0], box[1,1], box[2,0], box[2,1]
    l = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    V = l[0] * l[1] * l[2]
    if along_dimension == 'x':
        cross_area = l[1] * l[2]
    elif along_dimension == 'y':
        cross_area = l[0] * l[2]
    elif along_dimension == 'z':
        cross_area = l[0] * l[1]
        
    hist, bin_edges = np.histogram(single_coords, bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.
    DeltaV = cross_area * (bin_edges[1] - bin_edges[0])
    density = hist / DeltaV
    return np.column_stack((bin_centers, density))

def density_profile(bins, box, along_dimension):
    return lambda frame_t: density_profile0(frame_t, bins, box, along_dimension)
