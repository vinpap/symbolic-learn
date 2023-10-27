from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy


setup(
    name="symbolic-learn",
    version="0.1.4",
    ext_modules=[Extension('sblearn.compute', ['sblearn/compute.c']), 
                   Extension('sblearn.trees', ['sblearn/trees.c'])]
                            ,
    packages=find_packages(),
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

