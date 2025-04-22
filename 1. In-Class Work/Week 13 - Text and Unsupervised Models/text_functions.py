import numpy as np
import pandas as pd
import math
def calcKeyness(X, targets, minimum_threshold = 50, feature_names = None):
    """Calculates a keyness statistic for terms in a term-document matrix. 
    X should sparse matrix from CountVectorizer. Y should be a list of True False values"""
    target =  np.array(X[np.where(targets == True)].sum(axis=0)).flatten()
    baseline =  np.array(X[np.where(targets == False)].sum(axis=0)).flatten()
    keyness = []
    target_total = sum(target)
    baseline_total  =sum(baseline)
    norm = 1/ (baseline_total + target_total)
    for i in range(target.size):
        if feature_names is not None:
            term = feature_names[i]
        else:
            term = i
        target_count = target[i] if target[i] > 0 else norm
        baseline_count = baseline[i] if baseline[i] > 0 else norm
        if (baseline_count + target_count) < minimum_threshold: 
                continue
        target_prop = (target_count/target_total)
        baseline_prop = (baseline_count/baseline_total)
        stats = {'term': term,
                 'count_target' : int(target_count),
                 'count_baseline': int(baseline_count),
                 'oddsratio': math.log2( target_prop/baseline_prop)}
        keyness.append(stats)
    return pd.DataFrame(keyness).sort_values('oddsratio')

