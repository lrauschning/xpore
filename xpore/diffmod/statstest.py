import scipy.stats
import numpy as np
from typing import List

"""
This file implements some statistical tests, used both for prefiltering sites prior to GMM fitting to improve performance and for assigning significance after the fit.
It is called by worker threads in scripts/diffmod.py .
"""

# leon: best as I can tell, x is the one-hot condition label and y the estimated mean mod. rates

# data is stored in a dict that looks like this:
# (idx, pos, kmer) -> {'y': y, 'x': x, 'r': r, 'condition_names': condition_names_dummies, 'run_names': run_names_dummies, 'y_condition_names': condition_labels, 'y_run_names': run_labels}
# by default, in the call only the values are passed along

def t_test(data: List) -> float:
    labels = data['x'].astype(np.bool)
    means = data['y']

    # check for at most two conditions
    assert labels.shape[1] == 2

    # separate means by labels
    cond1means = means[labels[:,0]]
    cond2means = means[labels[:,1]]
    #cond2means = means[np.logical_not(labels)]

    # perform t test on data
    stats, _ = scipy.stats.ttest_ind(cond1means, cond2means)

    # compute two-sided significance. |labels| - 2 degrees of freedom
    return scipy.stats.t.sf(np.abs(stats), len(cond1means)+len(cond2means)-2)*2
    # equivalent, but sf can be more precise
    #(1 - scipy.stats.t.cdf(abs(stat), df)) * 2

def z_test(y1, y2, n1, n2):
    # two-tailed
    p1 = y1.mean()
    p2 = y2.mean()
    n1 = n1.mean().round()
    n2 = n2.mean().round()
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = (p1 - p2) / se
    return z, scipy.stats.norm.sf(abs(z))*2


def linear_test(data: List) -> float:
    pass

# dictionary to map config options to test implementations
METHODS_DICT= {"t_test":t_test, "t-test": t_test, "t": t_test,
               "linear_test": linear_test, "linear-test": linear_test, "l": linear_test,
               }

