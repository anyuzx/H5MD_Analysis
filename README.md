## Documentation

This module is used for analyzing LAMMPS H5MD format dump trajectory file. The valid trajectory file can be analyzed must contains position or velocity hdf5 group. And the path of the group is accessed through `['particles/all/position]`. The module right now can calculate **Mean Square Displacement** and **Intermediate Scattering Function**. However custom function can be easily defined as integrated into this module, as long as the it's function of configuration of position of particles at time $t$ and time $t+\tau$. And one is only interested in the average of quantity depends on $\tau$. For instance, **Mean Square Displacement** is one of the examples

![equation](http://www.sciweavers.org/upload/Tex2Img_1460784338/render.png)

$$g(\tau)=\frac{1}{N}\Bigg\langle \sum_{i=1}^{N}(\mathbf{r}_{i}(t+\tau) - \mathbf{r}_{i}(t))^{2}  \Bigg\rangle$$

The other example is **Intermediate Scattering Function**

$$F_{s}(k,\tau) = \frac{1}{N}\Bigg\langle \sum_{i=1}^{N}e^{i\mathbf{k}\cdot[\mathbf{r}_{i}(t+\tau) - \mathbf{r}_{i}(t)]} \Bigg\rangle$$

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

* `LammpsH5MD.cal_correlate(func_lst, t0freq, dtnumber=100, start=0, end=None, align=False, mode='log', size=[1])`

> Calculate the correlation quantity defined in `func_lst`.

> **Parameter**: func_lst: python list

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; t0freq: take initial timestep every t0freq frames

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; dtnumber: total number of time interval calculated

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; start: start frame subject to calculation

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; end: end frame subject to calculation. Default value: None. The last frame of file.

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; align: enable/disable trajectory alignment

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; mode: the method used to distribute the dt. Default value: log

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; size: size of quantity calculated. Each element of array specify the number of each function in func_lst. type: list/array.

> **Return**: python list of each quantity numpy array calculated by each function in func_lst. The first column of each quantity array is the dt array.

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
msd = traj.cal_correlate(msd.msd, 10, align=True)
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
We will explain every arguments in `LammpsH5MD.cal_correlate` by an example. Suppose we have a trajectory file which has total 20001 frames.

```python
LammpsH5MD.cal_correlate([msd.msd, isf.isf(4.0,26)], t0freq=10, dtnumber = 200, start = 10000, end = 15000, align = True, mode = 'log', size = [1,1])
```

* `func_lst`: this is a python list contains all the calculation module you want. The code above will use `msd.msd` and `isf.isf(4.0, 26)` calculate the quantity at the same time. No need to write two different codes and go through the trajectory file twice.
* `t0freq`: the frequency to take the initial time. The above example will take the initial frames `[10000, 10010, 10020, 10030, ..., 150000]`. 
* `start`: self-exlained. The code will analyze the data between frame `start` and frame `end`
* `end`: self-explained. If not specified, the code will analyze to the last frame of file.
* `align`: enable/disable trajectory alignment. The alignment is done using [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)
* `mode`: Two options: `log` and `linear`. The method used to sample the dt. In the above example, since `start=10000` and `end=15000`, we maximum dt is `start-end = 5000`. The `log` means that dt array is an array with `dtnumber=100` length in the range of 0 and 5000 such that the interval between each element is logrithm separated. `linear` means that element in dt array is separated uniformly. `log` mode is useful when log scale on time scale is needed.
* `size`: This is the list/array corresponds to `func_lst`. Each element of `size` should be an integer and is the dimension of qunatity returned by the corresponding function in `func_lst`. `msd.msd` only return one result, then the corresponding number in `size` should be one. If we have another function `msd.msd_2` which return both the mean suqare displacement of the sytem and the mean square displacement of the center of mass, then we should have `size=[2,1]` where 2 means our function return two quantities.

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
WRITE:
    - isf_k4.0.txt
        id: 2
    - g1.txt:
        id: 1
ARGS:
    t0freq: 10
    start: 10000
    end: Null
    align: True
    mode: log
    size: [1, 1]
```

#### Keywords
* `FILENAME`: Specify the path to the trajectory file
* `COMPUTE`: Specify the quantity computed. Give the name of function and arguments if necessary. Also assign a unique ID to each compute.
* `WRITE`: Specify the name of output file you want to use. ID corresponds to the `COMPUTE`.
* `ARGS`: Specify the arguments parsed to `LammpsH5MD.cal_correlate`. See above.
