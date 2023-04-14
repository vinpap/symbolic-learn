from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


"""setup(
    ext_modules = cythonize(["sblearn/*.pyx"], 
                            annotate=False,
                            compiler_directives={'profile': True, 
                                                 'boundscheck': False, 
                                                 'wraparound': False, 
                                                 'cdivision': True,
                                                 'binding': False,
                                                 'initializedcheck': False
                            }),
    install_requires=['Cython', 
                      'joblib', 
                      'numpy', 
                      'pandas', 
                      'scikit_learn', 
                      'setuptools', 
                      'sympy'
      ],
    include_dirs=[numpy.get_include()]
)"""


setup(
    ext_modules = [Extension('sblearn.compute', ['sblearn/compute.c']), 
                   Extension('sblearn.trees', ['sblearn/trees.c'])]
                            ,
    install_requires=['Cython', 
                      'joblib', 
                      'numpy', 
                      'pandas', 
                      'scikit_learn', 
                      'setuptools', 
                      'sympy'
      ],
    include_dirs=[numpy.get_include()]
)
