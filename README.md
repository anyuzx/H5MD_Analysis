# H5MD Analysis

Exploratory Python/Cython tools for analyzing LAMMPS H5MD trajectory files.

The package centers on `h5md_analysis.LammpsH5MD`, a small trajectory reader and
analysis driver. It loads particle positions from an H5MD file and evaluates
one-time or two-time observables supplied as Python callables.

## Requirements

- Python `>=3.14`
- Runtime dependencies from `pyproject.toml`: NumPy, SciPy, h5py, PyYAML, and
  tqdm
- Build dependencies: setuptools, wheel, Cython, NumPy, and a C/C++ compiler
  with OpenMP support for the compiled kernels

The editable install builds Cython extensions under `h5md_analysis._core`.

## Install

From the repository root:

```bash
python -m pip install -e .
```

Verify the package import:

```bash
python -c "from h5md_analysis import LammpsH5MD; print(LammpsH5MD.__name__)"
```

## Input Files

`LammpsH5MD.load()` expects the trajectory file to contain the LAMMPS H5MD
datasets used by this repo:

- `particles/all/position/value`
- `particles/all/box/edges/value`
- `particles/all/species`
- `particles/all/mol/value`

The code intentionally fails at the missing dataset if one of these assumptions
is not met.

## Command-Line Usage

After installation, run a YAML parameter file with:

```bash
h5md-analysis examples/parameter_example.txt
```

Useful options:

```bash
h5md-analysis -q examples/parameter_example.txt
h5md-analysis -l analysis.log examples/parameter_example.txt
```

- `-q` or `--quite`: suppress screen output
- `-l LOGFILE` or `--log LOGFILE`: write progress output to a log file

The YAML parameter file supports:

- `FILE`: input H5MD trajectory path
- `COMPUTE`: observable names, ids, and optional `args`
- `WRITE`: output filenames matched to compute ids
- `ARGS_TWOTIME`: keyword arguments for `LammpsH5MD.cal_twotime`
- `ARGS_ONETIME`: keyword arguments for `LammpsH5MD.cal_onetime`

Output paths are interpreted relative to the directory where the command is run.
Two-time outputs support text and `.pkl`; `.npy` is only supported for one-time
outputs.

## YAML Example

```yaml
FILE: my_h5md_traj.h5
COMPUTE:
    - isf:
        id: 1
        args:
            wave_vector: 5.0
            class_number: 26
    - msd.g1:
        id: 2
        args: {}
    - cmap:
        id: 3
        args:
            cutoff: 2.0
WRITE:
    - isf_k5.txt:
        id: 1
    - msd_g1.txt:
        id: 2
    - cmap.npy:
        id: 3
ARGS_TWOTIME:
    t0freq: 100
    dtnumber: 100
    start: 0
    end: Null
    align: 0
    mode: log
ARGS_ONETIME:
    tfreq: 1000
    start: 0
    end: Null
    align: 0
    reduce: sum
```

For factory-style observables such as `msd.g1`, use `args: {}` when the default
arguments should be bound.

## Python API

Two-time calculations receive functions of `frame_t1, frame_t2`:

```python
from h5md_analysis import LammpsH5MD
from h5md_analysis.calculations import isf, msd

traj = LammpsH5MD()
traj.load("my_h5md_traj.h5")

data = traj.cal_twotime(
    [
        msd.g1(),
        isf.isf(wave_vector=5.0, class_number=26),
    ],
    t0freq=100,
    dtnumber=100,
    start=0,
    align=0,
    mode="log",
)
```

One-time calculations receive functions of `frame_t`:

```python
from h5md_analysis.calculations import contactmap, distmap

data = traj.cal_onetime(
    [
        contactmap.contactmap(cutoff=2.0),
        distmap.distmap,
    ],
    tfreq=1000,
    start=0,
    align=0,
    reduce="sum",
)
```

Both methods return dictionaries keyed by the callable objects passed in.

## Available Observables

The CLI registry in `h5md_analysis.analysis_tool` exposes these names.

Two-time observables:

- `msd.g1`, `msd.g2`, `msd.g3`
- `ree_correlation`
- `isf`
- `cmapevolution`

One-time observables:

- `cmap`
- `rdf`
- `rdp`, `rdp_atom`
- `sdp`, `sdp_square`, `sdp_hist_square`, `sdp_hist_region`
- `ps`
- `dmap`, `dmap.square`
- `loop_gyration_tensor`, `type_gyration_tensor`, `gyration_tensor`
- `loop_orientation`

Additional calculation modules, including structure-factor routines, are
available from `h5md_analysis.calculations` for direct Python use.

## Project Layout

```text
src/h5md_analysis/
  trajectory.py        # LammpsH5MD trajectory reader and analysis driver
  analysis_tool.py     # h5md-analysis CLI implementation
  calculations/        # Python observable functions
  _core/               # Cython kernels
examples/              # YAML parameter examples
docs/                  # project notes
```

## Build Artifacts

The repo tracks source files only for Cython kernels. Generated C/C++ files,
compiled extensions, bytecode, egg-info, and build directories are ignored and
should not be committed.
