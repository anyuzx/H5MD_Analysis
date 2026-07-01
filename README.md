# H5MD Analysis

Exploratory Python/Cython tools for analyzing LAMMPS H5MD trajectory files.

The main class is `h5md_analysis.LammpsH5MD`. It reads H5MD trajectories and
evaluates one-time or two-time observables supplied as small calculation
functions.

## Install

Use Python 3.14 or newer.

```bash
python -m pip install -e .
```

The editable install builds the Cython extensions under
`h5md_analysis._core`.

## Layout

```text
src/h5md_analysis/
  trajectory.py        # LammpsH5MD trajectory reader and analysis driver
  analysis_tool.py     # CLI implementation
  calculations/        # Python observable functions
  _core/               # Cython kernels
examples/              # YAML parameter examples
docs/                  # project notes
```

Generated C/C++ files, compiled extensions, bytecode, and build directories are
not tracked.

## Python API

```python
from h5md_analysis import LammpsH5MD
from h5md_analysis.calculations import msd, isf

traj = LammpsH5MD()
traj.load("my_h5md_traj.h5")

data = traj.cal_twotime(
    [msd.g1(), isf.isf(wave_vector=4.0, class_number=26)],
    t0freq=10,
    start=0,
    align=0,
)
```

Calculation functions are ordinary callables:

- one-time functions receive `frame_t`
- two-time functions receive `frame_t1, frame_t2`

Factories such as `msd.g1(...)`, `isf.isf(...)`, and `contactmap.contactmap(...)`
return callables with parameters bound.

## CLI

After installation, run a YAML parameter file with:

```bash
h5md-analysis examples/parameter_example.txt
```

The parameter format supports:

- `FILE`: input H5MD trajectory path
- `COMPUTE`: observable names and optional arguments
- `WRITE`: output filenames mapped by compute id
- `ARGS_TWOTIME`: arguments passed to `LammpsH5MD.cal_twotime`
- `ARGS_ONETIME`: arguments passed to `LammpsH5MD.cal_onetime`

Supported CLI observable names are defined in
`h5md_analysis.analysis_tool.func_name_dic`.

## Built-in Observable Families

One-time observables include contact maps, distance maps, radial density
profiles, radial distribution functions, subchain distance profiles, structure
factors, and gyration/loop shape metrics.

Two-time observables include mean-squared displacement, intermediate scattering
functions, end-to-end correlation, and contact evolution.
