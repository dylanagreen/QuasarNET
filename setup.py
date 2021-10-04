#!/usr/bin/env python
import os, sys, glob, re
from setuptools import setup, find_packages

def _get_version():
    line = open('py/quasarnet/_version.py').readline().strip()
    m = re.match("__version__\s*=\s*'(.*)'", line)
    if m is None:
        print('ERROR: Unable to parse version from: {}'.format(line))
        version = 'unknown'
    else:
        version = m.groups()[0]

scripts = glob.glob('bin/*')

description = "CNN for quasar classification and redshifting"

setup(name="quasarnet",
      version=_get_version(),
      description=description,
      url="https://github.com/dylanagreen/QuasarNET",
      author="Dylan Green, James Farr, Nicolas Busca et al",
      author_email='dylanag@uci.edu',
      packages=['quasarnet'],
      package_dir = {'': 'py'},
      install_requires=['scipy','numpy',
          'fitsio','h5py','tensorflow'],
      #test_suite='picca.test.test_cor',
      scripts = scripts
      )

