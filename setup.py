from setuptools import setup, find_packages

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

from setup_scripts.precalc_matrices_compression import decompress_t_xy, decompress_t_z

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        decompress_t_xy()
        decompress_t_z()
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        decompress_t_xy()
        decompress_t_z()
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION



setup(
    name='MOFT',
    version='1.1.0',
     packages=find_packages(
        # All keyword arguments below are optional:
        where='.',  # '.' by default
        include=['LGTools*'],
    ),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    requires=['numpy', 'tqdm', 'matplotlib', 'scipy', 'sympy']
)