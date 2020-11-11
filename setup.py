"""
Setup of for domain-invariant representation learning
"""
from setuptools import setup

setup(name='dirl_core',
      description='Domain Invariant Representation Learning',
      author='Ajay Tanwani',
      author_email='aktanwani@gmail.com',
      package_dir={'':'src'},
      packages=['dirl_core', 'model_builders', 'dataset_builders'],
)
