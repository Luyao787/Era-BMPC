"""
Setup for sim package
"""
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['utils', 'motion_control', 'behavior_planner'],
    package_dir={'': 'src', '': 'src', '': 'src'},
)

setup(**d)