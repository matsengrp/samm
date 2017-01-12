#!/usr/bin/env scons

# Simulate data under various different settings and fit models

# Packages

import json
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

env = Environment(ENV=os.environ, 
                  NREPS = GetOption('nreps'))


# Set up state
base = {'nreps': env['NREPS']}

# Simulation methods
#sim_methods = ['survival_data', 'mutation_data']
sim_methods = ['mutation_data']

# Estimation methods
est_methods = ['null_model', 'survival_model']

# Potential nests: number of germlines, number of taxa from germline,
# frequency of mutation from germline

# Nests for replicates

name = 'output'
nest = SConsWrap(Nest(base_dict=base), '_'+name, alias_environment=env)

nest.add(
    'simulation_methods',
    sim_methods)

nest.add(
    'replicate',
    range(env['NREPS']),
    label_func='{:02d}'.format)

# Set the seed to be the replicate number.
nest.add(
    'seed',
    lambda c: [c['replicate']],
    create_dir=False)

nest.add_aggregate('per_rep', list)

# Targets for simulating fake data

@nest.add_target_with_env(env)
def generate(env, outdir, c):
    cmd = ['python simulate_from_sampled_gls.py',
           'simulate',
           '--seed',
           c['seed'],
           '--output_file ${TARGETS[0]}',
           '--output_genes ${TARGETS[1]}']
    return env.Command(
        [join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
        [],
        ' '.join(map(str, cmd)))

# Nest for model fitting

nest.add(
    'estimation_method',
    est_methods)

nest.add_aggregate('per_model_rep', list)

@nest.add_target_with_env(env)
def fit(env, outdir, c):
    cmd = 'echo $SOURCES ' + c['estimation_method'] + ' > $TARGET'

    model_fit = env.Command(
        join(outdir, c['estimation_method']+'-fit.csv'),
        [c['generate'][0], c['generate'][1]],
        cmd)
    c['per_model_rep'].append(model_fit)

# Aggregate over different fitting methods

# Aggregate over all replicates

# Plot results?

# working on it...

