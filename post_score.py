import scipy.stats as st
from math import *
from time import *

# constants
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
        #return z_table(0.999)

# scores range from 0 to 1

def score_post(post_time, pos_votes, neg_votes):
    # novelty
    cur_time = time()
    age = cur_time - post_time # seconds
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
    
    score = novelty * NOVELTY_WEIGHT + quality * QUALITY_WEIGHT
    return score

for i in range (0, 10):
    print(str(5 * 10 ** i) + ", " + str(2 * 10 ** i) + ": " + str(score_post(time(), 5 * 10 ** i, 2 * 10 ** i)))
