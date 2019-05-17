from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(name = 'navsim',
      author = 'Alby Musaelian',
      packages = ['navsim'],
      ext_modules = cythonize([
        "navsim/util.pyx"
      ]),
      include_dirs=[np.get_include()],
      install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-image"
      ])
