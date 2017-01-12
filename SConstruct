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

nest.add_aggregate('overall', list)

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

## Nest for model fitting
#
#nest.add_aggregate('per_rep', list)
#
#nest.add(
#    'estimation_method',
#    est_methods)
#
#@nest.add_target_with_env(env)
#def fit(env, outdir, c):
#    # TODO: remove dummy code and put in model fitting code
#    # using c['estimation_method'] for whatever model
#    # fitting code we have
#    cmd = 'echo $SOURCES > $TARGET'
#
#    model_fit = env.Command(
#        join(outdir, 'fit.csv'),
#        [c['generate'][0], c['generate'][1]],
#        cmd)
#    c['per_rep'].append(model_fit)
#
## Aggregate over different fitting methods
#
#nest.pop('estimation_method')
#
#@nest.add_target_with_env(env)
#def calculate_per_rep_score(env, outdir, c):
#    # TODO: remove dummy code
#    cmd = 'echo $SOURCES > $TARGET'
#    predictions = env.Command(
#        join(outdir, 'per_replicate_predictions.csv'),
#        c['per_rep'],
#        cmd)
#    c['overall'].append(predictions)
#
#
## Aggregate over all replicates
#
#nest.pop('replicate')
#
#@nest.add_target_with_env(env)
#def calculate_overall_scores(env, outdir, c):
#    # TODO: remove dummy code
#    return env.Command(join(outdir, 'predictions.csv'),
#                       c['overall'],
#                       'echo $SOURCES > $TARGET')
#
#
## Plot results?

