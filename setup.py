#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for MOFF
@author: Wei He
@email: whe3@mdanderson.org
"""

from setuptools import setup
import re
from os import path


def main():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md')) as f:
         long_description = f.read()
    #version = re.search('^__version__\s*=\s*"(.*)"',open('bin/MOFF').read(), re.M).group(1)
    setup(name='MOFF',
          version='1.1.0',
          author='Wei He',
          author_email='whe3@mdanderson.org',
          description="Modular prediction of off-target effects for CRISPR/Cas9 system",
          long_description=long_description,
          long_description_content_type='text/markdown',
          packages=['MOFF'],
          package_dir={'MOFF':'MOFF'},
          package_data={'MOFF':['StaticFiles/*']},
          url='https://github.com/MDhewei/MOFF',
          scripts=['bin/MOFF'],
          install_requires=[
                  'numpy',
                  'pandas',
                  'matplotlib',
                  'sklearn',
                  'argparse',
                  'seaborn'],
          zip_safe = True
        )
if __name__ == '__main__':
    main()
