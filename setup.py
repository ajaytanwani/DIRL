"""
Setup of for domain-invariant robot learning
"""
from setuptools import setup

setup(name='dirl_core',
      description='Domain Invariant Robot Learning',
      package_dir={'':'src'},
      packages=['dirl_core', 'model_builders', 'dataset_builders'],
)
