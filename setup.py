from setuptools import setup, find_packages
from Cython.Build import cythonize
import os

# Use Semantic Versioning, http://semver.org/
version_info = (0, 1, 12, '')
__version__ = '%d.%d.%d%s' % version_info


setup(name='ABRPlotting',
      version=__version__,
      description='Plotting package for manislab abrs',
      url='http://github.com/pbmanis/ABRPlotting',
      author='Paul B. Manis',
      author_email='pmanis@med.unc.edu',
      license='MIT',
      packages=find_packages(include=['src*']),
      zip_safe=False,
      entry_points={
          'console_scripts': [
               'abrp=src.plotABRs:main',
          ]
      },
      classifiers = [
             "Programming Language :: Python :: 3.8+",
             "Development Status ::  Beta",
             "Environment :: Console",
             "Intended Audience :: Neuroscientists",
             "License :: MIT",
             "Operating System :: OS Independent",
             "Topic :: Scientific Software :: Tools :: Python Modules",
             "Topic :: Data Processing :: Neuroscience",
             ],
    )
      