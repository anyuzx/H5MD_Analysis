# Agent Instructions

This repo contains exploratory Python/Cython analysis code for LAMMPS H5MD
trajectory files.

Main entry points:
- `LammpsH5MD.py` reads H5MD files and runs one-time or two-time calculations.
- `AnalysisTool.py` parses YAML-style parameter files and writes requested
  outputs.
- Top-level modules such as `msd.py`, `isf.py`, `rdf.py`, and `contactmap.py`
  define calculation functions.
- `src/*.pyx` contains Cython kernels built through `setup.py`.

## Coding Style

This repo is for exploratory data analysis and research code, not production software.

## Default style

Write code for clarity, directness, and easy debugging.

Prefer:
- simple, linear code
- explicit intermediate variables
- transparent assumptions
- fail-fast behavior
- minimal necessary checks

Avoid unless explicitly requested:
- defensive programming
- broad try/except
- classes or heavy abstraction
- generic utility layers
- handling many hypothetical edge cases
- silent fixes, fallbacks, or auto-cleaning

## Error philosophy

Do not hide errors.

If assumptions are violated, let the code fail clearly at the relevant step.
Use only a few targeted checks for important assumptions, such as required columns, merge keys, or missing IDs.

## Scope

Solve the specific task described.
Do not generalize the code into a reusable framework unless asked.

## Project maintenance

- Inspect the existing analysis modules before adding a new calculation.
- Prefer a small callable that matches the existing `cal_onetime` or
  `cal_twotime` patterns over adding new orchestration code.
- Keep H5MD dataset assumptions explicit, especially paths under
  `particles/all/...`.
- Avoid new dependencies unless explicitly requested.
- Keep generated trajectories, large analysis outputs, `.pyc`, and
  `__pycache__` out of commits.
- If changing a Cython kernel or `setup.py`, rebuild the extension and run a
  small import or smoke check for the affected module.

## Output preference

Write code like a careful researcher working in a notebook, not like a backend engineer building a service.
Keep it short, inspectable, and easy to modify.

## Markdown math convention

For Markdown documents, use single dollar signs for inline math, such as `$R(s)$`, and double dollar signs for block equations. Do not use `\(...\)` for inline math or `\[...\]` for block equations in new or edited Markdown prose.
