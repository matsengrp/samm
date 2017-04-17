import sys
import argparse
import os
import os.path
import csv
import pickle
import logging as log

import numpy as np
import scipy.stats

from submotif_feature_generator import SubmotifFeatureGenerator
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--motif-len',
        type=int,
        help='motif length',
        default=5)
    parser.add_argument('--compare-folder',
        type=str,
        help='folder of models to compare',
        default='_output')
    parser.add_argument('--model-types',
        type=str,
        help='list of model types to compare, comma separated',
        default='survival_L,shmulate')
    parser.add_argument('--reps',
        type=int,
        help='num replicates',
        default=2)
    parser.add_argument('--model-fit-file',
        type=str,
        help='fitted model file name',
        default='fitted.pkl')
    parser.add_argument('--theta-file',
        type=str,
        help='true theta file name',
        default='true_theta.pkl')
    parser.add_argument('--out-file',
        type=str,
        help='out file name with all the comparisons',
        default='out.txt')

    parser.set_defaults()
    args = parser.parse_args()
    args.model_types = args.model_types.split(",")
    return args

def load_true_model(file_name):
    with open(file_name, "rb") as f:
        true_theta, probability_matrix = pickle.load(f)
        true_theta -= np.median(true_theta)
        return true_theta, probability_matrix

def load_fitted_model(file_name, model_type, is_per_target):
    with open(file_name, "rb") as f:
        if model_type.startswith("survival"):
            survival_fits = pickle.load(f)
            # right now assume we just have one fit
            fitted_theta, fitted_prob_vector = survival_fits
        elif model_type.startswith("basic"):
            fitted_theta, fitted_prob_vector = pickle.load(f)
            fitted_theta -= np.median(fitted_theta)
        elif model_type == "shmulate":
            fitted_theta, (_, fitted_prob_vector) = pickle.load(f)
            # correct for baseline
            fitted_theta -= np.median(fitted_theta)
        else:
            raise ValueError()
    return fitted_theta, fitted_prob_vector

class ModelStats:
    def __init__(self):
        self.theta_spearmanr = []
        self.theta_pearsonr = []
        self.theta_err1 = [] # L1 distance
        self.theta_err2 = [] # L2 distance

        self.prob_spearmanr = []
        self.prob_pearsonr = []
        self.prob_err1 = [] # L1 distance??
        self.prob_err2 = [] # L2 distance??

    def append(self, model, true_model):
        assert(model[0].size == true_model[0].size)
        model_theta = np.reshape(model[0], true_model[0].shape)

        theta_mask = np.where(true_model[0] != -np.inf)
        fitted_theta = model_theta[theta_mask].flatten()
        true_theta = true_model[0][theta_mask].flatten()
        s_corr = scipy.stats.spearmanr(fitted_theta, true_theta)
        self.theta_spearmanr.append(s_corr[0])
        p_corr = scipy.stats.pearsonr(fitted_theta, true_theta)
        self.theta_pearsonr.append(p_corr[0])
        theta_err1 = np.linalg.norm(true_theta - fitted_theta, ord=1)
        self.theta_err1.append(theta_err1)
        theta_err2 = np.linalg.norm(true_theta - fitted_theta, ord=2)/np.linalg.norm(fitted_theta, ord=2)
        self.theta_err2.append(theta_err2)

        # if true_model[1] is not None:
        #     prob_mask = np.where(true_model[1] != 0)[0]
        #     fitted_probs = np.array(model[1])[prob_mask].flatten()
        #     true_probs = true_model[1][prob_mask].flatten()
        #     s_corr = scipy.stats.spearmanr(fitted_probs, true_probs)
        #     self.prob_spearmanr.append(s_corr[0])
        #     p_corr = scipy.stats.pearsonr(fitted_probs, true_probs)
        #     self.prob_pearsonr.append(p_corr[0])
        #     prob_err1 = np.linalg.norm(true_probs - fitted_probs, ord=1)
        #     self.prob_err1.append(prob_err1)
        #     prob_err2 = np.linalg.norm(true_probs - fitted_probs, ord=2)
        #     self.prob_err2.append(prob_err2)

    def print_summary(self):
        def get_mean_var(vals):
            return "%.4g (%.4g)" % (np.mean(vals), np.sqrt(np.var(vals)))

        if len(self.theta_spearmanr):
            print "Theta Spearman\t", get_mean_var(self.theta_spearmanr)
            print "Theta Pearson\t", get_mean_var(self.theta_pearsonr)
            print "Theta L1 Err\t", get_mean_var(self.theta_err1)
            print "Theta L2 Err\t", get_mean_var(self.theta_err2)
        if len(self.prob_spearmanr):
            print "Prob Spearman\t", get_mean_var(self.prob_spearmanr)
            print "Prob Pearson\t", get_mean_var(self.prob_pearsonr)
            print "Prob L1 Err\t", get_mean_var(self.prob_err1)
            print "Prob L2 Err\t", get_mean_var(self.prob_err2)

def main(args=sys.argv[1:]):
    args = parse_args()

    model_stats = {m: ModelStats() for m in args.model_types}
    for i in range(args.reps):
        theta_file_name = os.path.join(args.compare_folder, "%02d" % i, args.theta_file)
        true_model = load_true_model(theta_file_name)
        num_cols = true_model[0].shape[1]
        is_per_target = num_cols == NUM_NUCLEOTIDES + 1

        for model_type in args.model_types:
            fitted_file_name = os.path.join(args.compare_folder, "%02d" % i, model_type, args.model_fit_file)
            fitted_model = load_fitted_model(fitted_file_name, model_type, is_per_target)
            model_stats[model_type].append(fitted_model, true_model)

    for model_type, stats in model_stats.iteritems():
        print "==== Model type: %s ====" % model_type
        stats.print_summary()

if __name__ == "__main__":
    main(sys.argv[1:])
