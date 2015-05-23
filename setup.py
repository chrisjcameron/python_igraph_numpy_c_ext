#python setup.py build_ext --inplace -I/usr/local/include

from distutils.core import setup, Extension
import numpy.distutils.misc_util

setup(
    ext_modules=[Extension(
        "ignp_fun",
        ["ignp_fun.c"],
        library_dirs=['/usr/local/lib'],
        libraries=['igraph'],
        extra_compile_args=["-O3"],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
        )],
)

