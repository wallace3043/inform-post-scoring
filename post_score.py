'''
Created by Wallace Long for Inform
Created Apr 22, 2020

Outputs a score from (0, 1000000) for a post, given its time of posting (datetime),
number of positive votes, and number of negative votes.
'''

import scipy.stats as st
from math import *
from time import *

# constants
MAX_SCORE = 1000000
NOVELTY_WEIGHT = 0.6
QUALITY_WEIGHT = 1 - NOVELTY_WEIGHT
DECAY_SHIFT = 3.5
DAY_IN_SECS = 60 * 60 * 24

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper
    
@memoize
def z_table(precision):
    return st.norm.ppf(precision)

def z_score(n):
    if n <= 10:
        return 0
    else:
        num_nines = log(n)
        return z_table(1 - e ** - num_nines)

# scores range from 0 to 1000000

def score_post(post_time, pos_votes, neg_votes):
    # novelty
    cur_time = time()
    age = cur_time - post_time # in seconds
    novelty = 1 / (1 + e ** (age / DAY_IN_SECS - DECAY_SHIFT))
    
    # quality
    n = pos_votes + neg_votes
    if (n == 0):
        quality = 0.5
    else :
        p = pos_votes / n
        z = z_score(n)
        max_err = z * sqrt(p * (1 - p) / n)
        quality = max(0, p - z * max_err)
    
    score = MAX_SCORE * novelty * NOVELTY_WEIGHT + quality * QUALITY_WEIGHT
    return floor(score)
