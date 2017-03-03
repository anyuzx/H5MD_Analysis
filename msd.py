import numpy as np

# function g1, g2, g3
# g1: MSD of whole system (internal motion + global motion)
# g2: MSD of internal motion (in the reference of center of mass of the system)
# g3: MSD of the center of mass of the system
# g1 =\frac{1}{N}\Bigg\langle \int_{i=1}^{i=N}(r_{i}(t) - r_{i}(0))^{2} \Bigg\rangle
# g2 =\frac{1}{N}\Bigg\langle \int_{i=1}^{i=N}(r_{i}^{0}(t) - r_{i}^{0}(0))^{2} \Bigg\rangle
# g3 = \Bigg\langle (r_{com}(t) - r_{com}(0))^2 \Bigg\rangle


def g10(frame_t1, frame_t2, index=None):
    if index is None:
        return np.sum(np.mean(np.power(frame_t1 - frame_t2, 2), axis=0))
    else:
        if not isinstance(index, (list, tuple, np.ndarray)):
            raise ValueError("argument [index] provided is not a list/tuple/array object.\n")

        result = []
        for atom_index in index:
            atom_index = np.array(atom_index, dtype=np.int) - 1
            temp = np.sum(np.mean(np.power(frame_t1[atom_index] - frame_t2[atom_index], 2), axis=0))
            result.append(temp)
        return np.array(result)

def g1(index=None):
    return lambda frame_t1, frame_t2: g10(frame_t1, frame_t2, index)

def g20(frame_t1, frame_t2, index=None):
    com_t1 = np.mean(frame_t1, axis=0)
    com_t2 = np.mean(frame_t2, axis=0)
    frame_t1_com = frame_t1 - com_t1
    frame_t2_com = frame_t2 - com_t2

    if index is None:
        return np.sum(np.mean(np.power(frame_t1_com - frame_t2_com, 2), axis=0))
    else:
        if not isinstance(index, (list, tuple, np.ndarray)):
            raise ValueError("argument [index] provided is not a list/tuple/array object.\n")

        result = []
        for atom_index in index:
            atom_index = np.array(atom_index, dtype=np.int) - 1
            temp = np.sum(np.mean(np.power(frame_t1_com[atom_index] - frame_t2_com[atom_index], 2), axis=0))
            result.append(temp)
        return np.array(result)

def g2(index=None):
    return lambda frame_t1, frame_t2: g20(frame_t1, frame_t2, index)

def g3(frame_t1, frame_t2):
    com_t1 = np.mean(frame_t1, axis=0)
    com_t2 = np.mean(frame_t2, axis=0)

    return np.sum(np.power(com_t1 - com_t2, 2))
