from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


numpy_include = numpy.get_include()

ext_modules = [Extension('core._contactmap', ['src/_contactmap.pyx'], include_dirs = [numpy_include]),
               Extension('core._sdp',['src/_sdp.pyx'], include_dirs = [numpy_include]),
               Extension('core._sdp_hist',['src/_sdp_hist.pyx'], include_dirs = [numpy_include]),
               Extension('core._sdp_hist_region',['src/_sdp_hist_region.pyx'], include_dirs = [numpy_include]),
               Extension('core._ps',['src/_ps.pyx'], include_dirs = [numpy_include]),
               Extension('core._distmap',['src/_distmap.pyx'], include_dirs = [numpy_include]),
               Extension('core._contactevolution',['src/_contactevolution.pyx'],include_dirs = [numpy_include]),
               Extension('core._loop_gyration_tensor',['src/_loop_gyration_tensor.pyx'],include_dirs = [numpy_include]),
               Extension('core._type_gyration_tensor',['src/_type_gyration_tensor.pyx'],include_dirs = [numpy_include]),
               Extension('core._gyration_tensor', ['src/_gyration_tensor.pyx'], include_dirs = [numpy_include]),\
               Extension('core._loop_orientation', ['src/_loop_orientation.pyx'], include_dirs = [numpy_include]),\
               Extension(name="core._pdist_pbc",
                sources=["src/_pdist_pbc.pyx"],
                library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'],
                include_dirs=['/usr/include/'],
                extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp']),\
               Extension(name="core._sk_debye",
                sources=["src/_sk_debye.pyx"],
                library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'],
                include_dirs=['/usr/include/', numpy_include],
                extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp'],
                language='c++'),\
              Extension(name="core._sk_lebedev",
                sources=["src/_sk_lebedev.pyx"],
                library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'],
                include_dirs=['/usr/include/', numpy_include],
                extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp'],
                language='c++'),\
              Extension(name="core._sk_direct",
                sources=["src/_sk_direct.pyx"],
                library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'],
                include_dirs=['/usr/include/', numpy_include],
                extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp'],
                language='c++'),\
              Extension(name="core._rdf",
                sources=["src/_rdf.pyx"],
                library_dirs=['/usr/lib/gcc/x86_64-pc-linux-gnu/11.1.0/'],
                include_dirs=['/usr/include/', numpy_include],
                extra_compile_args=['-O3', '-ffast-math', '-fopenmp'],
                extra_link_args=['-fopenmp'],
                language='c++')
  ]

setup(
  ext_modules = cythonize(
    ext_modules,
    compiler_directives={'language_level': '3'},
  ),
)
