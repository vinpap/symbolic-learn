from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy


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

