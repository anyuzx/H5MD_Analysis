# Documentation

## LammpsH5MD

This module is used for analyzing LAMMPS H5MD format dump trajectory file. The valid trajectory file can be analyzed must contains position or velocity HDF5 group. And the path of the group is accessed through `['particles/all/position]`. The main function of this module is to calculate one-time quantity or two-time correlation quantity from a given LAMMPS H5MD format trajectory file. The examples for one-time quantity includes density profile, contact map, distance map, radial distribution functions, structural factors, etc. Two-time correlation quantity include mean squared displacement, dynamic scattering function, etc. The module is designed such that custom function can be defined as separate plug-in module and easily be used. The only requirement for any custom function is that it's a function of position of particles at time $t$ for one-time quantity or $t$ and $t+\tau$ for two-time quantity. The examples of two-time correlation quantity for *mean squared displacement* is given below,

$$
\mathrm{MSD} = \frac{1}{N}\bigg\langle \sum_{i}^{N} (\boldsymbol{r}_i(t) - \boldsymbol{r}_i(0))^2 \bigg\rangle
$$

and *intermediate scattering function*,

$$
F_s(\boldsymbol{k},t)=\frac{1}{N}\bigg\langle \sum_{i}^{N} e^{i \boldsymbol{k}(\boldsymbol{r}_i(t) - \boldsymbol{r}_i(0))} \bigg\rangle
$$

### The current built-in function to calculate
**One-time**
- Contact map
- Density profile
- Distance map
- Radius of gyration tensor
- Radial distribution function
- Structural factors
- End-to-end distance

**Two-time**
- Mean squared displacement
- Intermediate scattering function

### How to use

The main program is `LammpsH5MD.py`. It defines the class `LammpsH5MD` which is used to read and process H5MD file. It has several class members.

* `LammpsH5MD.load(fname)`

> Load trajectory file

> **Parameter**: fname: path of H5MD file

> **Return**: None

* `LammpsH5MD.get_framenumber()`

> Get the total number of frames stored in trajectory

> **Parameter**: None.

> **Return**: None. The value is stored in LammpsH5MD.frame_number

* `LammpsH5MD.get_frame(i)`

> Give the position of particles of ith frame

> **Parameter**: i: int type. E.g i=0 means the first frame

> **Return**: position_array: [N, 3] array of particle position of ith frame.

* `LammpsH5MD.cal_twotime(func_lst, t0freq, dtnumber=100, start=0, end=None, align=False, mode='log')`

> Calculate the two time quantity defined in `func_lst`. Two time quantity is the quantity determined by system state at two different time.

> **Parameter**: func_lst: python list

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t0freq: take initial timestep every t0freq frames

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dtnumber: total number of time interval calculated

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; start: start frame subject to calculation

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; end: end frame subject to calculation. Default value: None. The last frame of file.

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; align: enable/disable trajectory alignment. Specify the index of reference snapshot to enable it. E.g. align=0. Default value: False

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mode: the method used to distribute the dt. Default value: log

> **Return**: a python dictionary whose keys are the object of function and values are the quantity associated to that function.

* `LammpsH5MD.cal_onetime(func_lst, tfreq=1, start=0, end=None, align=False, reduce='sum')`

> Calculate the one time quantity defined in `func_lst`. One time quantity is the quantity determined by system state at two different time.

> **Parameter**: func_lst: python list

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; tfreq: calculate the quantity every tfreq frames

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; start: start frame subject to calculation

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; end: end frame subject to calculation. Default value: None. The last frame of file.

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; align: enable/disable trajectory alignment. Specify the index of reference snapshot to enable it. E.g. align=0. Default value: False

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; reduce: the method used to summarize the quantity. Default value: 'sum'. Add quantity calculated at different timesteps together.

> **Return**: a python dictionary whose keys are the object of function and values are the quantity associated to that function.

* `LammpsH5MD.extract_traj(foutname, stride, start=0, end=None)`

> Extract subset of full trajectory and write to a new H5MD formatted file.

> **Parameter**: foutname: path/name of file you want to write

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; stride: Extract frame every this many number of frames

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; start: start index of frames subject to extraction. Default: the first frame of original trajectory file.

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; end: end index of frames subject to extraction. Default: The last frame of original trajectory file.

> **Return**: None. Invoke this method, a new file will be written on disk.

* `LammpsH5MD.info()`

> Print out the avaiable information about the loaded file

> **Parameter**: None

> **Return**: None

### Tutorial

#### General Use
Suppose we want to calculate the **Mean Square Displacement** of trajectory file `my_h5md_traj.h5`. The following code can do this

``` python
import numpy as np
import LammpsH5MD
import msd
import sys

traj = LammpsH5MD.LammpsH5MD()
traj.load('my_h5md_traj.h5')
msd = traj.cal_twotime(msd.g1, t0freq=10, start=0, align=0)
```

####  Calculation Function Module
`msd` in the above code is a module which defines the function which actually calculate the quantity. Such function can have parameters, but eventually should only be a function of system configuration at time $t$ and at time $t+\tau$. Not let's look at what module `msd` actually contains

``` python
import numpy as np

def msd(frame_t1, frame_t2):
    return np.sum(np.mean(np.power(frame_t1, frame_t2, 2), axis=0))
```

It's very simple. function `msd` calculate the displacement between frame at time `t1` and frame at time `t2`. This quantity is calculated by many times in class `LammpsH5MD.cal_correlate()` and average over different initial timesmteps. We can define any function which is a **correlation** function and easily integrate into our code. For instance, let's look at another example which calculate **Intermediate Scattering Function**, we use Lebedev quadrature to integrate(average) over wave vector $\mathbf{k}$. The code is the following:

``` python
import numpy as np
import Lebedev.Lebedev as Lebedev # import function to create Lebedev grid point and weights

def isf0(frame_t1, frame_t2, wave_vector, class_number):
    grid = Lebedev(class_number)
    temp = np.inner(frame_t1, frame_t2, grid[:, 0:3])
    temp = np.sum(np.cos(wave_vector * temp) * grid[:, -1], axis=1)
    return np.mean(temp)

# define isf lambda function, this is the actual functin passed to class LammpsH5MD routine
def isf(wave_vector, class_number):
    return lambda frame_t1, frame_t2: isf0(frame_t1, frame_t2, wave_vector, class_number)
```

Notice that we define a lambda function above because our function depends on $\mathbf{k}$ and how many grid points we want to use in Lebedev quadrature. So we need to define a lambda function to pass a parameterized function to our `LammpsH5MD.cal_correlate()`.

#### Explanination of Arguments
We will explain every arguments in `LammpsH5MD.cal_twotime` by an example. Suppose we have a trajectory file which has total 20001 frames.

```python
LammpsH5MD.cal_twotime([msd.msd, isf.isf(4.0,26)], t0freq=10, dtnumber = 200, start = 10000, end = 15000, align = 0, mode = 'log')
```

* `func_lst`: this is a python list contains all the calculation module you want. The code above will use `msd.msd` and `isf.isf(4.0, 26)` calculate the quantity at the same time. No need to write two different codes and go through the trajectory file twice.
* `t0freq`: the frequency to take the initial time. The above example will take the initial frames `[10000, 10010, 10020, 10030, ..., 150000]`.
* `start`: self-exlained. The code will analyze the data between frame `start` and frame `end`
* `end`: self-explained. If not specified, the code will analyze to the last frame of file.
* `align`: enable/disable trajectory alignment. Speicfy the index of frame you want to use as reference.  The alignment is done using [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)
* `mode`: Two options: `log` and `linear`. The method used to sample the dt. In the above example, since `start=10000` and `end=15000`, we maximum dt is `start-end = 5000`. The `log` means that dt array is an array with `dtnumber=100` length in the range of 0 and 5000 such that the interval between each element is logrithm separated. `linear` means that element in dt array is separated uniformly. `log` mode is useful when log scale on time scale is needed.

#### Use parameter file
We can use a parameter file to parse the arguments to `LammpsH5MD`. The parameter file use `YAML` syntax. For instance:

```yaml
FILE: my_test_h5md.h5
COMPUTE:
    - msd.g1:
        id: 1
    - isf:
        id: 2
        args:
            wave_vector: 4.0
            class_number: 26
    - cmap:
        id: 3
        args:
            cutoff: 2.0
WRITE:
    - isf_k4.0.txt:
        id: 2
    - g1.txt:
        id: 1
    - cmap.npy:
        id: 3
ARGS_TWOTIME:
    t0freq: 10
    start: 10000
    end: Null
    align: 10000
    mode: log
ARGS_ONETIME:
    tfreq: 1000
    start: 0
    end: Null
    align: 0
```

##### Keywords
* `FILE`: Specify the path to the trajectory file
* `COMPUTE`: Specify the quantity computed. Give the name of function and arguments if necessary. Also assign a unique ID to each compute.
* `WRITE`: Specify the name of output file you want to use. ID corresponds to the `COMPUTE`.
* `ARGS_TWOTIME`: Specify the arguments parsed to `LammpsH5MD.cal_twotime`. See above.
* `ARGS_ONETIME`: Specify the arguments parsed to `LammpsH5MD.cal_onetime`. See above.

#### Compute Modules List
* `isf`: calculate the self intermediate scattering function
* `msd`: calculate `g1`, `g2` and `g3` part of mean square displacement of system
* `cmap`: calculate contact map given certain threshold determining contact
* `rdp`: calculate radial denstiy profile
* `sdp`: calculate subchain distance profile
* `dmap`: calculate distance map

