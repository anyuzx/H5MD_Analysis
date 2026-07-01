import numpy as np

def ree_corr0(frame_t1, frame_t2, index1=None, index2=None):
	if index2 is None and index1 is not None:
		assert type(index1) == int
		ree1 = frame_t1[-1] - frame_t1[index1]
		ree2 = frame_t2[-1] - frame_t2[index1]
		corr = np.sum(ree1 * ree2)
	elif index2 is not None and index1 is None:
		assert type(index2) == int
		ree1 = frame_t1[index2] - frame_t1[0]
		ree2 = frame_t2[index2] - frame_t2[0]
		corr = np.sum(ree1 * ree2)
	elif index2 is not None and index1 is not None:
		assert type(index2) == int
		assert type(index1) == int
		ree1 = frame_t1[index2] - frame_t1[index1]
		ree2 = frame_t2[index2] - frame_t2[index1]
		corr = np.sum(ree1 * ree2)
	else:
		ree1 = frame_t1[-1] - frame_t1[0]
		ree2 = frame_t2[-1] - frame_t2[0]
		corr = np.sum(ree1 * ree2)
	
	return corr

def ree_corr(index1=None, index2=None):
	return lambda frame_t1, frame_t2: ree_corr0(frame_t1, frame_t2, index1, index2)