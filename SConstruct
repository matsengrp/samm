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
                  OUTPUT_NAME = GetOption('output_name'))


# Set up state
base = {'nreps': env['NREPS'],
        'output_name': env['OUTPUT_NAME']}

# Potential nests: simulation methods, estimation methods, number of germlines,
# number of taxa from germline, frequency of mutation from germline

nest = SConsWrap(Nest(base_dict=base), '_'+env['OUTPUT_NAME'], alias_environment=env)

# Nest for simulation methods
sim_methods = [
    #'survival_big', #This is still too big to run our data on.
    'survival_mini',
    'shmulate',
]

nest.add(
    'simulation_methods',
    sim_methods)

# Nest for fitting models
model_options = [
    'survival',
    'basic',
]

nest.add(
    'model_options',
    model_options)

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
               '--output-true-theta ${TARGETS[0]}',
               '--output-file ${TARGETS[1]}',
               '--output-genes ${TARGETS[2]}']
        return env.Command(
            [join(outdir, 'true_theta.pkl'), join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
            [],
            ' '.join(map(str, cmd)))
    elif c['simulation_methods'] == "survival_mini":
        cmd = ['python simulate_from_survival.py',
               '--seed',
               c['seed'],
               '--n-taxa',
               5,
               '--n-germlines',
               50,
               '--motif-len',
               3,
               '--random-gene-len',
               30,
               '--min-censor-time',
               3.0,
               '--ratio-nonzero',
               0.2,
               '--output-true-theta ${TARGETS[0]}',
               '--output-file ${TARGETS[1]}',
               '--output-genes ${TARGETS[2]}']
        return env.Command(
            [join(outdir, 'true_theta.pkl'), join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
            [],
            ' '.join(map(str, cmd)))
    elif c['simulation_methods'] == "survival_big":
        cmd = ['python simulate_from_survival.py',
               '--seed',
               c['seed'],
               '--n-taxa',
               10,
               '--n-germlines',
               50,
               '--motif-len',
               5,
               '--random-gene-len',
               200,
               '--min-censor-time',
               1.0,
               '--ratio-nonzero',
               0.1,
               '--output-true-theta ${TARGETS[0]}',
               '--output-file ${TARGETS[1]}',
               '--output-genes ${TARGETS[2]}']
        return env.Command(
            [join(outdir, 'true_theta.pkl'), join(outdir, 'seqs.csv'), join(outdir, 'genes.csv')],
            [],
            ' '.join(map(str, cmd)))
    else:
        raise NotImplementedError()

## Future nests

# Nest for model fitting
@nest.add_target_with_env(env)
def fit_context_model(env, outdir, c):
    if c["simulation_methods"] == "survival_mini":
        motif_len = 3
    else:
        motif_len = 5

    if c["model_options"] == "survival":
        cmd = ['python fit_context_model.py',
               '--seed',
               c['seed'],
               '--motif-len',
               motif_len,
               '--penalty-params',
               "0.05",
               '--num-threads',
               10,
               '--input-file ${SOURCES[0]}',
               '--input-genes ${SOURCES[1]}',
               '--theta-file ${SOURCES[2]}',
               '--log-file ${TARGETS[0]}',
               '--out-file ${TARGETS[1]}']
        return env.Command(
            [join(outdir, 'context_log.txt'), join(outdir, 'context_log.pkl')],
            [join(outdir, 'seqs.csv'), join(outdir, 'genes.csv'), join(outdir, 'true_theta.pkl')],
            ' '.join(map(str, cmd)))
    else:
        cmd = ['python fit_basic_model.py',
               '--seed',
               c['seed'],
               '--motif-len',
               motif_len,
               '--input-file ${SOURCES[0]}',
               '--input-genes ${SOURCES[1]}',
               '--theta-file ${SOURCES[2]}',
               '--prop-file ${TARGETS[0]}',
               '--log-file ${TARGETS[1]}']
        return env.Command(
            [join(outdir, 'proportions.pkl'), join(outdir, 'log.txt')],
            [join(outdir, 'seqs.csv'), join(outdir, 'genes.csv'), join(outdir, 'true_theta.pkl')],
            ' '.join(map(str, cmd)))


# Aggregate over different fitting methods

# Aggregate over all replicates

# Plot results
