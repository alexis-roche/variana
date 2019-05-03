#!/usr/bin/env python
version = '0.1dev'

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup


def configuration(parent_package='', top_path=None):
    config = Configuration('variana', parent_package, top_path)
    #config.set_options(
    #    ignore_setup_xxx_py=True,
    #    assume_default_configuration=True,
    #    delegate_options_to_subpackages=True,
    #    quiet=True,
    #)
    config.add_data_dir('tests')
    #config.add_subpackage('variational_sampling')
    config.add_extension('_utils', sources=['_utils.c'])   
    return config


def setup_package():
    setup(
        configuration=configuration,
        name='variana',
        version=version,
        maintainer='Alexis Roche',
        maintainer_email='alexis.roche@gmail.com',
        description='Gaussian probability distribution approximation via variational sampling',
        url='http://www.scipy.org',
        license='BSD',
        #install_requires=['numpy >= 1.0.2',],
    )
    return

if __name__ == '__main__':
    setup_package()
