import sys
from setuptools import setup, find_packages

try:
    import numpy as np
except ImportError:
    print("Need numpy for installation")
    sys.exit(1)

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

setup(
    name='Dfit',
    author='Sören von Bülow',
    version='0.2',
    license='GPLv3',
    install_requires=['numpy', 'scipy', 'numba', 'MDAnalysis>=0.19.0', 'scipy>=0.19', 'matplotlib','progressbar2'],
    packages=find_packages(),
    include_dirs=[numpy_include],
)
