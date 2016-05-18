import numpy as np
import h5py
import sys
import time

__all__ = ['LammpsH5MD']

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
# funtion used to calculate mean and variance one-pass algorithm
class OnlineVariance:
    """
    Welford's algorithm to computes the sample mean, variance in a stream
    """

    def __init__(self, dim, length):
        self.mean, self.var, self.S = np.zeros((length, dim)), np.zeros((length, dim)), np.zeros((length, dim))
        self.n = np.zeros(length)

    def stream(self, data_point, position):
        self.n[position] += 1.0
        self.delta = data_point - self.mean[position]
        self.mean[position] += self.delta / float(self.n[position])
        self.S[position] += self.delta * (data_point - self.mean[position])
        self.var[position] = self.S[position] / (self.n[position] - 1.0)

        return self.mean, self.var
# =====================================================================

class LammpsH5MD:
    def __init__(self):
        self.file = self.filename = None
        self.frame_number = None
        self.atoms_number = None

    def load(self, finname):
        # load the trajectory
        try:
            self.filename = finname
            self.file = h5py.File(finname, 'r')
        except:
            raise

    def get_framenumber(self):
        # get the total number of snapshots stored in file
        try:
            self.frame_number = self.file['particles/all/position/value'].shape[0]
        except:
            raise

    def get_atomnumber(self):
        # get the total number of atoms/particles stored in file
        try:
            self.atoms_number = self.file['particles/all/position/value'].shape[1]
        except:
            raise

    def get_frame(self,t):
        # get the frame provided the index of snapshot
        return self.file['particles/all/position/value'][t]

    def cal_twotime(self, func_lst, t0freq=10, dtnumber=100, start=0, end=None,
                      align=False, mode='log'):
        # This method calculate any two-time quantity. Like MSD, ISF ...
        # And return the full list of data
        #
        # explain each argument:
        #   func_lst: list of functions to calculate the quantity.
        #       can specify multiple functions. E.g [msd,isf]
        #       if only one function, no need to use [ ]
        #   t0freq: take initial start time t0 for every this many frames
        #   dtnumber: length of dt_lst
        #   start: the index of first frame you want to analyze. Default value=0
        #   end: the index of last frame you want to analyze. Default value:
        #                                                     last frame of file
        #   align: enable/disable the trajectory alignment. If enable, specify the
        #           index of reference snapshot. Ex. align=0. Default value: False
        #
        #   mode: specify the method to distribute the dt. Default value: log
        #       log: make the dt list logrithm
        #       linear: linear distributedt
        #
        #   return: quantity list calculated. the order of list is the same as
        #       the func_lst.

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
        # E.g corr = {msd: []
        #             isf: []}
        twotime = {}

        for f in func_lst:
            twotime[f] = []

        # initial configuration. used for alignment if enabled
        if align is not False:
            frame_start = self.file['particles/all/position/value'][align]
        t_start = time.time()

        for t0 in t0_lst:
            sys.stdout.write('Initial time {} analyzed.\n'.format(t0))
            sys.stdout.flush()
            for index, dt in enumerate(dt_lst[np.where(dt_lst <= (end-t0-1))]):
                if align is not False:
                    frame_t1 = self.get_frame(t0+dt)
                    frame_t2 = self.get_frame(t0)
                    frame_t1 = optimal_rotate(frame_t1, frame_start)
                    frame_t2 = optimal_rotate(frame_t2, frame_start)
                else:
                    frame_t1 = self.get_frame(t0+dt)
                    frame_t2 = self.get_frame(t0)
                for func in func_lst:
                    twotime_temp = func(frame_t1, frame_t2) # t1 > t2
                    twotime[func].append([t0, t0+dt, twotime_temp]) # store the full information

        t_end = time.time()
        t_cost = t_end - t_start
        sys.stdout.write('\nTotal time cost: {:.2f} mins\n'.format(t_cost/60.0))

        # convert the python array to numpy array
        for key in twotime:
            twotime[key] = np.array(twotime[key])

        return twotime # return the dictionary

    def cal_onetime(self, func_lst, tfreq=1, start=0, end=None, align=False, reduce='sum'):
        # This method calculate any static(one-time) quantity. Like Energy ...
        # And return the full list of data
        #
        # explain each argument:
        #   func_lst: list of functions to calculate the quantity.
        #       can specify multiple functions. E.g [msd,isf]
        #       if only one function, no need to use [ ]
        #   tfreq: calculate the quantity for every this many frames
        #   start: the index of first frame you want to analyze. Default value=0
        #   end: the index of last frame you want to analyze. Default value:
        #                                                     last frame of file
        #   align: enable/disable the trajectory alignment. If enable, specify the
        #           index of reference snapshot. E.g. align=0. Default value: False
        #   reduce: reduce the result. E.g. reduce=sum will sum all the quantity together
        #
        #   return: a python dictionary whose keys are the object of function and values
        #            are the quantity associated to that function

        # all get_timesteps() if no self.frame_number
        if not self.frame_number:
            self.get_framenumber()

        if end == None:
            end = self.frame_number
        else:
            assert type(end) == int

        assert type(t0freq) == int
        assert type(start) == int
        if type(func_lst) != list:
            func_lst = [func_lst]

        # create the initial time array.
        t_lst = np.arange(start, end, tfreq)

        # start the calculation
        # initialize the quantity dictionary
        # E.g corr = {msd: []
        #             isf: []}
        onetime = {}

        for f in func_lst:
            onetime[f] = []
        onetime['time list'] = t_lst

        # initial configuration. used for alignment if enabled
        if align is not False:
            frame_start = self.file['particles/all/position/value'][align]

        t_start = time.time()

        for t in t_lst:
            sys.stdout.write('Initial time {} analyzed.\n'.format(t))
            sys.stdout.flush()
            if align is not False:
                frame_t = self.get_frame(t)
                frame_t = optimal_rotate(frame_t, frame_start)
            else:
                frame_t = self.get_frame(t)
            for func in func_lst:
                onetime_temp = func(frame_t)
                if reduce == 'sum':
                    try:
                        onetime[func] += onetime_temp
                    except ValueError:
                        onetime[func] = onetime_temp
                elif reduce == 'mean':
                    try:
                        temp[func].stream(onetime_temp, 0)
                        onetime[func] = temp[func].mean
                    except NameError:
                        temp = {}
                        temp[func] = OnlineVariance(1, 1)
                        temp[func].stream(onetime_temp, 0)
                        onetime[func] = temp[func].mean
                elif reduce == 'var':
                    try:
                        temp[func].stream(onetime_temp, 0)
                        onetime[func] = temp[func].var
                    except NameError:
                        temp = {}
                        temp[func] = OnlineVariance(1, 1)
                        temp[func].stream(onetime_temp, 0)
                        onetime[func] = temp[func].var
                elif reduce == 'None' or reduce == 'none':
                    onetime[func].append(onetime_temp)

        t_end = time.time()
        t_cost = t_end - t_start
        sys.stdout.write('\nTotal time cost: {:.2f} mins\n'.format(t_cost/60.0))

        # convert the python array to numpy array if necessary
        for key in onetime:
            onetime[key] = np.array(onetime[key])

        return onetime # return the dictionary

    def info(self):
        sys.stdout.write('File Loaded: {}\n'.format(self.filename))
        sys.stdout.write('Toal Number of Frames: {}\n'.format(self.frame_number))

    def extract_traj(self, foutname, stride, start=0, end=None):
        # all get_timesteps() if no self.frame_number
        if not self.frame_number:
            self.get_framenumber()

        if not self.atoms_number:
            self.get_atomnumber()

        if end == None:
            end = self.frame_number
        else:
            assert type(end) == int

        sys.stdout.write('Initialize the new H5MD file\n')
        new_traj = h5py.File(foutname, 'w') # create a new file

        # first get the dimension of dataset in original trajectory file
        nframes = self.frame_number
        natoms = self.atoms_number

        new_nframes = np.arange(nframes)[start:end:stride].shape[0]

        # copy h5md, observables, parameters group
        self.file.copy('/h5md', new_traj)
        self.file.copy('/observables', new_traj)
        self.file.copy('/parameters', new_traj)

        # create particles/all group
        new_traj.create_group('/particles/all')

        # create all the group in particles/all/
        new_traj.create_group('/particles/all/position')
        new_traj.create_group('/particles/all/box/edges')
        new_traj.create_group('/particles/all/velocity')

        # create dataset with new dimension position and velocity
        for key in ['position','velocity']:
            new_traj.create_dataset('particles/all/' + key + '/time', (new_nframes, ), dtype='float64')
            new_traj.create_dataset('particles/all/' + key + '/step', (new_nframes, ), dtype='int32')
            new_traj.create_dataset('particles/all/' + key + '/value', (new_nframes, natoms, 3), dtype='float64')

        # create box dataset
        new_traj.create_dataset('particles/all/box/edges/time', (new_nframes, ), dtype='float64')
        new_traj.create_dataset('particles/all/box/edges/step', (new_nframes, ), dtype='int32')
        new_traj.create_dataset('particles/all/box/edges/value', (new_nframes, 3), dtype='float64')

        # loop throught the original trajectory file
        # store the frames into our new file

        t_start = time.time()
        for index, t in enumerate(np.arange(nframes)[start:end:stride]):
            sys.stdout.write('Process timestep {}\n'.format(t))
            sys.stdout.flush()
            for key in ['position', 'velocity']:
                new_traj['particles/all/' + key + '/time'][index] = np.copy(self.file['particles/all/' + key + '/time'][t])
                new_traj['particles/all/' + key + '/step'][index] = np.copy(self.file['particles/all/' + key + '/step'][t])
                new_traj['particles/all/' + key + '/value'][index] = np.copy(self.file['particles/all/' + key + '/value'][t])

            new_traj['particles/all/box/edges/time'][index] = np.copy(self.file['particles/all/box/edges/time'][t])
            new_traj['particles/all/box/edges/step'][index] = np.copy(self.file['particles/all/box/edges/step'][t])
            new_traj['particles/all/box/edges/value'][index] = np.copy(self.file['particles/all/box/edges/value'][t])


        new_traj.flush()
        new_traj.close()
        self.file.close()
        t_end = time.time()
        t_cost = t_end - t_start
        sys.stdout.write('\nTotal time cost: {:.2f} mins\n'.format(t_cost/60.0))
        sys.stdout.flush()

        return
