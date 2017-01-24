#!/usr/bin/env scons

# Simulate data under various different settings and fit models

# Packages

import os
from os.path import join
from nestly.scons import SConsWrap
from nestly import Nest
from SCons.Script import Environment, Command, AddOption

# Command line options

AddOption('--nreps',
          dest='nreps',
          default=2,
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
                  OUTPUT_NAME = GetOption('output_name'))


# Set up state
base = {'nreps': env['NREPS'],
        'output_name': env['OUTPUT_NAME']}

# Potential nests: simulation methods, estimation methods, number of germlines,
# number of taxa from germline, frequency of mutation from germline

nest = SConsWrap(Nest(base_dict=base), '_'+env['OUTPUT_NAME'], alias_environment=env)

# Nest for simulation methods
sim_methods = ['shmulate', 'survival']

nest.add(
    'simulation_methods',
    sim_methods)

# Nest for replicates

nest.add(
    'replicate',
    range(env['NREPS']),
    label_func='{:02d}'.format)

# Set the seed to be the replicate number.
nest.add(
    'seed',
    lambda c: [c['replicate']],
    create_dir=False)

# Targets for simulating fake data

@nest.add_target_with_env(env)
def generate(env, outdir, c):
    if c['simulation_methods'] == "shmulate":
        cmd = ['python simulate_from_sampled_gls.py',
               'simulate',
               '--seed',
               c['seed'],
               '--output-file ${TARGETS[0]}',
               '--output-genes ${TARGETS[1]}']
    elif c['simulation_methods'] == "survival":
        cmd = ['python simulate_from_survival.py',
               '--seed',
               c['seed'],
               '--output-file ${TARGETS[0]}',
               '--output-genes ${TARGETS[1]}']
    return env.Command(
        [join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
        [],
        ' '.join(map(str, cmd)))

## Future nests

# Nest for model fitting
@nest.add_target_with_env(env)
def fit_context_model(env, outdir, c):
    cmd = ['python fit_context_model.py',
           '--seed',
           c['seed'],
           '--input-file ${SOURCES[0]}',
           '--input-genes ${SOURCES[1]}',
           '--log-file ${TARGETS[0]}',
           '--out-file ${TARGETS[1]}']
    return env.Command(
        [join(outdir, 'context_log.txt'), join(outdir, 'context_log.pkl')],
        [join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
        ' '.join(map(str, cmd)))

# Aggregate over different fitting methods

# Aggregate over all replicates

# Plot results
