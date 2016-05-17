from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


ext_modules = [Extension('core/_contactmap', ['src/_contactmap.pyx'], include_dirs = [numpy.get_include()])]

setup(
  ext_modules = cythonize(ext_modules),
)
