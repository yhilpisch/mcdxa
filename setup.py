#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get version from mcdxa/__init__.py
about = {}
with io.open(os.path.join(here, 'mcdxa', '__init__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

# Read the long description from README.md
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mcdxa',
    version=about.get('__version__', '0.0.0'),
    author='The Python Quants GmbH',
    author_email='',
    description='Python package for pricing European and American options via Monte Carlo simulation and analytic models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yhilpisch/mcdxa',
    packages=find_packages(exclude=['tests', 'scripts']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Office/Business :: Financial :: Investment',
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
    ],
    include_package_data=True,
    zip_safe=False,
)
