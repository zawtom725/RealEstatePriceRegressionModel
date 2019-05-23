from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['google-cloud', 'numpy', 'scikit-learn', 'tensorflow']

setup(
    name='trainer',
    version='0.0.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Generic example trainer package with dependencies.')
