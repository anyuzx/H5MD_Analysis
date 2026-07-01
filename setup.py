from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


CORE_DIR = Path('src/h5md_analysis/_core')
numpy_include = numpy.get_include()


def core_extension(name, source, *, include_numpy=True, language=None, openmp=False):
    include_dirs = []
    if include_numpy:
        include_dirs.append(numpy_include)
    if openmp:
        include_dirs.append('/usr/include/')

    return Extension(
        f'h5md_analysis._core.{name}',
        [str(CORE_DIR / source)],
        include_dirs=include_dirs,
        library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'] if openmp else [],
        extra_compile_args=['-O3', '-ffast-math', '-fopenmp'] if openmp else [],
        extra_link_args=['-fopenmp'] if openmp else [],
        language=language,
    )


ext_modules = [
    core_extension('_contactmap', '_contactmap.pyx'),
    core_extension('_sdp', '_sdp.pyx'),
    core_extension('_sdp_hist', '_sdp_hist.pyx'),
    core_extension('_sdp_hist_region', '_sdp_hist_region.pyx'),
    core_extension('_ps', '_ps.pyx'),
    core_extension('_distmap', '_distmap.pyx'),
    core_extension('_contactevolution', '_contactevolution.pyx'),
    core_extension('_loop_gyration_tensor', '_loop_gyration_tensor.pyx'),
    core_extension('_type_gyration_tensor', '_type_gyration_tensor.pyx'),
    core_extension('_gyration_tensor', '_gyration_tensor.pyx'),
    core_extension('_loop_orientation', '_loop_orientation.pyx'),
    core_extension('_pdist_pbc', '_pdist_pbc.pyx', include_numpy=False, openmp=True),
    core_extension('_sk_debye', '_sk_debye.pyx', language='c++', openmp=True),
    core_extension('_sk_lebedev', '_sk_lebedev.pyx', language='c++', openmp=True),
    core_extension('_sk_direct', '_sk_direct.pyx', language='c++', openmp=True),
    core_extension('_rdf', '_rdf.pyx', language='c++', openmp=True),
]


setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={'language_level': '3'},
    ),
)
