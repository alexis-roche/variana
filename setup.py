#!/usr/bin/env python

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

version = "0.1dev"

extensions = [
    Extension(
        name="variana._utils",
        sources=["variana/_utils.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="variana",
    version=version,
    maintainer="Alexis Roche",
    maintainer_email="alexis.roche@gmail.com",
    description="Gaussian probability distribution approximation via variational sampling",
    url="http://www.scipy.org",
    license="BSD",
    packages=find_packages(),
    package_data={"variana": ["tests/*"]},
    ext_modules=cythonize(extensions, language_level="3"),
)
