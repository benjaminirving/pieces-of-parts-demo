from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("additional.supervoxel",
              sources=["additional/supervoxel.pyx"],
              include_dirs=[numpy.get_include()])
]

setup(name="pieces-of-parts",
      version="0.1",
      description="Demo of pieces of parts to improve regional probability on supervoxels",
      author="Benjamin Irving ",
      url="birving.com",
      setup_requires=['Cython'],
      install_requires=['matplotlib', 'numpy', 'pandas', 'nibabel'],
      ext_modules=cythonize(extensions))

