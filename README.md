## Documentation

This module is used for analyzing LAMMPS H5MD format dump trajectory file. The valid trajectory file can be analyzed must contains position or velocity hdf5 group. And the path of the group is accessed through `['particles/all/position]`. The module right now can calculate **Mean Square Displacement** and **Intermediate Scattering Function**. However custom function can be easily defined as integrated into this module, as long as the it's function of configuration of position of particles at time `t` and time `t+\tau`. And one is only interested in the average of quantity depends on `\tau`. For instance, **Mean Square Displacement** is one of the examples

![](https://bgqomq-bn1305.files.1drv.com/y3ma_qrvcA0bpOrZB_oS8VaFSke_j64BCfFRwnHDoVSq-YJojYu3vwvp7iEHWE8m0AXgKiNospRy9rXePK2dOQ5LsSH7fSZUw2t-CO-FNtCLJyMuhQqywMS2xLlhhTwLa059X4_vcOB4uQFPInXw0gQ2zCG38esuLkNljDVSLZfHpI?width=256&height=55&cropmode=none)

The other example is **Intermediate Scattering Function**

![](https://bgqjbg-bn1305.files.1drv.com/y3m7LFJ8-PE_iAUza3je3v2LisGtKLlBrtNL8jXMYQAXbomg9Fgf83bZqW-wcREmTvoIv2c-K_UqCl2xeqV7sNnVkXMi1vgVXycDmgfjg8pY96K1dECVvC9RaY5Sk6gn5GAPEqaUg8hhjETnotdEgLwOebgqyAS_8K-1KF1QSCa7Vk?width=256&height=53&cropmode=none)

## How to use

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

* `LammpsH5MD.info()`

> Print out the avaiable information about the loaded file

> **Parameter**: None

> **Return**: None

## Tutorial

### General Use
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

###  Calculation Function Module
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

### Explanination of Arguments
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

### Use parameter file
We can use a parameter file to parse the arguments to `LammpsH5MD`. The parameter file use `YAML` syntax. For instance:

```
FILENAME: my_test_h5md.h5
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

#### Keywords
* `FILENAME`: Specify the path to the trajectory file
* `COMPUTE`: Specify the quantity computed. Give the name of function and arguments if necessary. Also assign a unique ID to each compute.
* `WRITE`: Specify the name of output file you want to use. ID corresponds to the `COMPUTE`.
* `ARGS_TWOTIME`: Specify the arguments parsed to `LammpsH5MD.cal_twotime`. See above.
* `ARGS_ONETIME`: Specify the arguments parsed to `LammpsH5MD.cal_onetime`. See above.
