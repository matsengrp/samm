#!/usr/bin/env scons

# Simulate data under various different settings and fit models

import os

from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
from SCons.Script import Environment
from SCons.Script import AddOption

# Command line options

AddOption('--nreps',
          dest='nreps',
          default=1,
          type='int',
          nargs=1,
          action='store',
          help='number of replicates')

AddOption('--output_name',
          dest='output_name',
          default='output',
          type='str',
          nargs=1,
          help='name of output directory')

env = Environment(ENV=os.environ,
                  NREPS = GetOption('nreps'),
                  OUTPUT_NAME = GetOption('output_name'),
                  CXX = 'clang++',
                  CXXFLAGS = '-std=c++11 -stdlib=libc++',
                  LINKFLAGS = '-stdlib=libc++')


mobeef_cpp_env = env.Clone()
mobeef_cpp_env.VariantDir('_build/mobeef', 'c/src', duplicate=0)
mobeef_cpp_env.Library(target='_build/mobeef/mobeef',
                      source=Glob('_build/mobeef/*.cpp'))

test_env = env.Clone()
test_env.VariantDir('_build/test', 'c/test', duplicate=0)
test_env.Append(CPPPATH=['c/src'])
test_env.Program(target='_build/test/test',
                 LIBPATH=['_build/mobeef'],
                 LIBS=['mobeef'],
                 source=Glob('_build/test/*.cpp'))

Export('env')

env.SConsignFile()

flag = 'simulation_section'
SConscript(flag + '/sconscript', exports=['flag'])

flag = 'simulated_shazam_vs_samm'
SConscript(flag + '/sconscript', exports=['flag'])

flag = 'multiple_mutations'
SConscript(flag + '/sconscript', exports=['flag'])

flag = 'imputed_ancestors_comparison'
SConscript(flag + '/sconscript', exports=['flag'])

flag = 'shazam_vs_samm'
SConscript(flag + '/sconscript', exports=['flag'])

flag = 'get_hedgehog'
SConscript(flag + '/sconscript', exports=['flag'])
