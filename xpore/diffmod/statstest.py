import scipy.stats
import numpy as np

"""
This file implements some statistical tests, used both for prefiltering sites prior to GMM fitting to improve performance and for assigning significance after the fit.
It is called by worker threads in scripts/diffmod.py .
"""

# dictionary to map config options to test implementations
METHODS_DICT= {"t_test":t_test, "t-test": t_test, "t": t_test,
               "linear_test": linear_test, "linear-test": linear_test, "l": linear_test,
               }

class StatsTest:
    def __init__(self,data):
        if self.__isok(data):
            self.data = [data['y'][data['x'][:,0]==1],data['y'][data['x'][:,1]==1]] 
        else:
            self.data = None

    def fit(self,method):
        pval = np.nan
        if self.data is not None:
            if method == 't-test':
                stats,_ = scipy.stats.ttest_ind(self.data[0], self.data[1])
                df = len(self.data[0]) + len(self.data[1]) - 2
                pval = scipy.stats.t.sf(np.abs(stats), df)*2 # also defined as 1 - cdf, but sf is sometimes more accurate. #(1 - scipy.stats.t.cdf(abs(stat), df)) * 
        return pval
    
    def __isok(self,data):
        return data['x'].shape[1] == 2 # Check if only two conditions.


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
    cond1means = means[labels]
    cond2means = means[np.logical_not(labels)]

    # perform t test on data
    stats, _ = scipy.stats.ttest_ind(cond1means, cond2means)

    # compute two-sided significance. |labels| - 2 degrees of freedom
    return scipy.stats.t.sf(np.abs(stats), len(labels)-2)*2
    # equivalent, but sf can be more precise
    #(1 - scipy.stats.t.cdf(abs(stat), df)) * 2

def linear_test(data: List) -> float:
    pass
