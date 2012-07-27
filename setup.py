#!/usr/bin/python

from distutils.core import setup, Extension
from glob import glob

lib_sources = glob('lib/*.c')
lib_sources.remove('lib/svm_loqo.c') # this is an alternate backend for SVM-Light; only
                                     # one of {svm_loqo.c,svm_hideo.c} may be compiled
                                     # with this extension.
lib_sources.remove('lib/svm_classify.c') # this file implements the "classify" binary;
                                         # don't include it, since it defines main()
                                         # again!

setup(name         = 'svmlight',
      version      = '0.4',
      description  = 'Interface to Thorsten Joachims\' SVM-Light, conversion of C-based interface by William Cauchois',
      author       = "Ceri Stagg",
      author_email = "ceristaggbusiness@gmail.com",
      url          = "https://github.com/KaitainCPS/trupysvmlight",
      long_description = open('README.md').read(),
      classifiers  = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      ext_modules = [
        Extension('svmlight', include_dirs = ['lib/'],
                  sources = lib_sources)
      ])
