import scipy.stats
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

"""
This file implements some statistical tests, used both for prefiltering sites prior to GMM fitting to improve performance and for assigning significance after the fit.
It is called by worker threads in scripts/diffmod.py .
"""

# leon: best as I can tell, x is the one-hot condition label and y the estimated mean mod. rates

# data is stored in a dict that looks like this:
# (idx, pos, kmer) -> {'y': y, 'x': x, 'r': r, 'condition_names': condition_names_dummies, 'run_names': run_names_dummies, 'y_condition_names': condition_labels, 'y_run_names': run_labels}
# by default, in the call only the values are passed along


@dataclass
class TestResult:
    diff: float
    pval: float
    esize: float
    formula: str

    def get_header(self) -> List[str]:
        return [f"diff_{formula}, p_{formula}", f"ES_{formula}"]

    def get_row(self) -> List[float]:
        return [self.diff, self.pval, self.esize]


def _separate_conds(labels, data) -> Tuple[np.array, np.array]:
    # check for at most two conditions
    assert labels.shape[1] == 2

    # separate data cols by labels
    c1data = data[labels[:,0]]
    c2data = data[labels[:,1]]
    #c2data = data[np.logical_not(labels)]

    return c1data, c2data


def t_test(data: Dict, model) -> TestResult:
    c1means, c2means = _separate_conds(data['x'].astype(np.bool), data['y'])

    # perform t test on data
    tstat, _ = scipy.stats.ttest_ind(c1means, c2means)

    # compute two-sided significance. |labels| - 2 degrees of freedom
    return TestResult(c1means - c2means, scipy.stats.t.sf(np.abs(tstat), len(c1means)+len(c2means)-2)*2, tstat, '')
    # equivalent, but sf can be more precise
    #(1 - scipy.stats.t.cdf(abs(stat), df)) * 2

#w = model.nodes['w'].expected()  # GK
#coverage = np.sum(model.nodes['y'].params['N'], axis=-1)  # GK => G # n_reads per group


def z_test(data: Dict, model) -> TestResult:
    print(data['x'], data['y'])
    
    # separate both means and cov data
    c1means, c2means = _separate_conds(data['x'].astype(np.bool), data['y'])
    c1covs, c2covs = _separate_conds(data['x'].astype(np.bool), data['y'])
    #TODO how to get cov data into there? stored in the model, not available in prefiltering???
    p1 = y1.mean()
    p2 = y2.mean()
    n1 = n1.mean().round()
    n2 = n2.mean().round()
    se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = (p1 - p2) / se
    # do a two-tailed test
    TestResult(c2means - c2means, scipy.stats.norm.sf(abs(z))*2, z, '')


def linear_test(data: List, model) -> TestResult:
    pass

# dictionary to map config options to test implementations
METHODS_DICT= {"t_test":t_test, "t-test": t_test, "t": t_test,
               "linear_test": linear_test, "linear-test": linear_test, "l": linear_test,
               }
PREFILT_METHODS_DICT = METHODS_DICT

