from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [Extension('core/_contactmap', ['src/_contactmap.pyx'], include_dirs = [numpy.get_include()]),
               Extension('core/_sdp',['src/_sdp.pyx'], include_dirs = [numpy.get_include()]),
               Extension('core/_distmap',['src/_distmap.pyx'], include_dirs = [numpy.get_include()]),
               Extension('core/_contactevolution',['src/_contactevolution.pyx'],include_dirs = [numpy.get_include()]),
               Extension('core/_loop_gyration_tensor',['src/_loop_gyration_tensor.pyx'],include_dirs = [numpy.get_include()]),
               Extension('core/_type_gyration_tensor',['src/_type_gyration_tensor.pyx'],include_dirs = [numpy.get_include()]),
               Extension('core/_gyration_tensor', ['src/_gyration_tensor.pyx'], include_dirs = [numpy.get_include()]),\
               Extension('core/_loop_orientation', ['src/_loop_orientation.pyx'], include_dirs = [numpy.get_include()])]

setup(
  ext_modules = cythonize(ext_modules),
)
