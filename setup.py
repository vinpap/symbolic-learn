from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize(["sblearn/*.pyx"], 
                            annotate=False,
                            compiler_directives={'profile': True, 
                                                 'boundscheck': False, 
                                                 'wraparound': False, 
                                                 'cdivision': True,
                                                 'binding': False,
                                                 'initializedcheck': False
                            }),
    include_dirs=[numpy.get_include()]
)