import numpy as np
import h5py
import sys
import time

# =====================================================================
# PRIVATE function
# ---------------------------------------------------------------------
# DEFINE OPTIMAL ROTATION FUNCTION
def optimal_rotate(P,Q):
    # P and Q are two sets of vectors
    P = np.matrix(P)
    Q = np.matrix(Q)

    assert P.shape == Q.shape

    Qc = np.mean(Q,axis=0)

    P = P - np.mean(P,axis=0)
    Q = Q - np.mean(Q,axis=0)

    # calculate covariance matrix A = (P^T)Q
    A = P.T * Q

    # SVD for matrix A
    V, S, Wt = np.linalg.svd(A)

    # correct rotation matrix to ensure a right-handed system if necessary
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:,-1] = -V[:,-1]

    # calculate the final rotation matrix U
    U = V * Wt

    return np.array(P * U + Qc)
# =====================================================================

# =====================================================================
# TEST TEST TEST
class OnlineVariance1:
    """
    Welford's algorithm to computes the sample mean, variance in a stream
    """

    def __init__(self, dim):
        if dim == 1:
            self.mean, self.var, self.S = 0,0,0
        else:
            self.mean, self.var, self.S = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        self.n = 0

    def stream(self, data_point):
        self.n += 1
        self.delta = data_point - self.mean
        self.mean += self.delta / float(self.n)
        self.S += self.delta * (data_point - self.mean)
        if self.n > 1:
            self.var = self.S / (self.n - 1)

    def stat(self):
        return self.mean, self.var

class OnlineVariance2:

    def __init__(self, dim):
        if dim == 1:
            self.mean, self.var, self.sumx, self.sumx2 = 0,0,0,0
        else:
            self.mean, self.var, self.sumx, self.sumx2 = np.zeros(dim), np.zeros(dim), np.zeros(dim), np.zeros(dim)
        self.n = 0

    def stream(self, data_point):
        self.n += 1
        self.sumx += data_point
        self.sumx2 += np.power(data_point, 2.0)

    def stat(self):
        self.mean = self.sumx / self.n
        self.var = (self.sumx2 - np.power(self.sumx, 2.0) / float(self.n)) / float(self.n)
        #self.var = self.sumx2 / self.n - (self.mean)**2.0
        return self.mean, self.var
# TEST TEST TEST
# =====================================================================

class LammpsH5MD:
    def __init__(self):
        self.file = self.filename = None
        self.frame_number = None

    def load(self, finname):
        try:
            self.filename = finname
            self.file = h5py.File(finname, 'r')
        except:
            raise

    def get_framenumber(self):
        try:
            self.frame_number = self.file['particles']['all']['position']\
                                         ['step'][:].shape[0]
        except:
            raise

    def get_frame(self,t):
        return self.file['particles']['all']['position']['value'][t]

    def cal_correlate(self, func_lst, t0freq=10, dtnumber=100, start=0, end=None,
                      align=False, mode='log', size=[1], variance=False):
        # explain each argument:
        #   func_lst: list of functions to calculate the quantity.
        #       can specify multiple functions. E.g [msd,isf]
        #       if only one function, no need to use [ ]
        #   t0freq: take initial start time t0 for every this many frames
        #   dtnumber: length of dt_lst
        #   start: the index of first frame you want to analyze. Default value=0
        #   end: the index of last frame you want to analyze. Default value:
        #                                                     last frame of file
        #   align: enable/disable the trajectory alignment
        #   mode: specify the method to distribute the dt. Default value: log
        #       log: make the dt list logrithm
        #       linear: linear distributed
        #   size: size of quantity calculated. type:array
        #       each element of array specify the number for each function in
        #       func_lst
        #   variance: enable/disable variance storing. If enabled, both \sum_{i}x_{i}
        #       and \sum_{i}x_{i}^2 are stored.
        #
        #   return: quantity list calculated. the order of list is the same as
        #       the func_lst. The firt column of each quantity array is the
        #       delta t array.

        # all get_timesteps() if no self.frame_number
        if not self.frame_number:
            self.get_framenumber()

        if end == None:
            end = self.frame_number
        else:
            assert type(end) == int

        assert type(t0freq) == int
        assert type(dtnumber) == int
        assert type(start) == int
        if type(func_lst) != list:
            func_lst = [func_lst]
        assert len(func_lst) == len(size)
        for item in size: assert type(item) == int

        # create the initial time array.
        t0_lst = np.arange(start, end, t0freq)
        # create the delta t array. Two possible modes
        # linear mode
        if mode == 'linear':
            dt_lst = np.unique(np.int_(np.linspace(0, end-start-1, dtnumber)))
        elif mode == 'log':
            dt_lst = np.unique(np.int_(np.power(10, np.linspace(0, np.log10(end-start-1), dtnumber))))
            dt_lst = np.insert(dt_lst, 0, 0)
        else:
            raise ValueError("Error: value of argument 'mode' not recognized...\n")

        # start the calculation
        # initialize the quantity dictionary
        # E.g corr = {msd: np.zeros((100,1))
        #             isf: np.zeros((100,1))}
        corr = {}
        if variance:
            corr_square = {}
        for f, s in zip(func_lst, size):
            corr[f] = np.zeros((len(dt_lst), s))
            if variance:
                corr_square[f] = np.zeros((len(dt_lst), s))

        # create a array storing the number of quantity calcualted
        # used for the average
        corr_count = np.zeros((len(dt_lst), 1))
        # initial configuration. used for alignment if enabled
        frame_start = self.file['particles']['all']['position']['value'][start]
        t_start = time.time()
        for t0 in t0_lst:
            sys.stdout.write('\rInitial time {} analyzed.'.format(t0))
            sys.stdout.flush()
            for index, dt in enumerate(dt_lst[np.where(dt_lst <= (end-t0-1))]):
                if align:
                    frame_t1 = self.get_frame(t0+dt)
                    frame_t2 = self.get_frame(t0)
                    frame_t1 = optimal_rotate(frame_t1, frame_start)
                    frame_t2 = optimal_rotate(frame_t2, frame_start)
                else:
                    frame_t1 = self.get_frame(t0+dt)
                    frame_t2 = self.get_frame(t0)
                for func in func_lst:
                    corr_temp = func(frame_t1, frame_t2) # t1 > t2
                    corr[func][index] += corr_temp
                    if variance:
                        corr_square[func][index] += np.power(corr_temp, 2.0)
                corr_count[index] += 1

        t_end = time.time()
        t_cost = t_end - t_start
        sys.stdout.write('\nTotal time cost: {:.2f} mins\n'.format(t_cost/60.0))
        corr_lst = []
        if variance:
            corr_variance_lst = []
        for key in corr:
            corr_lst.append(np.hstack((dt_lst.reshape((len(dt_lst), 1)), corr[key]/corr_count)))
            if variance:
                corr_variance_lst.append(np.hstack((dt_lst.reshape((len(dt_lst), 1)), \
                corr_square[key]/corr_count - np.power(corr[key]/corr_count, 2.0))))
        if variance:
            return corr_lst, corr_variance_lst
        else:
            return corr_lst

    def info(self):
        sys.stdout.write('File Loaded: {}\n'.format(self.filename))
        sys.stdout.write('Toal Number of Frames: {}\n'.format(self.frame_number))
