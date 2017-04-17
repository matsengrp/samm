import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
import csv
from common import *

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input-pkl',
        type=str,
        help='pickle file with theta values')
    parser.add_argument('--output',
        type=str,
        default='_output/out.pdf')

    args = parser.parse_args()

    return args

def main(args=sys.argv[1:]):

    args = parse_args()

    thetas = pickle.load(open(args.input_pkl, "r"))
    fit_theta = thetas[0]

    print "3mer", np.sum(np.abs(fit_theta[:64,0]) > ZERO_THRES)
    print "5mer", np.sum(np.abs(fit_theta[64: 64 + 1024,0]) > ZERO_THRES)
    print "7mer", np.sum(np.abs(fit_theta[64 + 1024:,0]) > ZERO_THRES)

    f, axarr = plt.subplots(3, sharex=True, figsize=(6,8))
    axarr[0].hist(fit_theta[:64,0], 10)
    axarr[0].set_title("3-mer theta values")
    axarr[1].hist(fit_theta[64:64+1024,0], 50)
    axarr[1].set_title("5-mer theta values")
    # axarr[2].hist(fit_theta[64+1024:], 500)
    # axarr[2].set_title("7-mer theta values")
    axarr[2].hist(fit_theta[64+1024:,0], 500)
    axarr[2].set_yscale('log')
    axarr[2].set_title("7-mer theta values (Log y)")
    plt.savefig(args.output)

if __name__ == "__main__":
    main(sys.argv[1:])
