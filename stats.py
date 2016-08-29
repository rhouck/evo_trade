import hashlib

import numpy as np
from functools import partial
from deap import tools


def nan_fit_ratio(x):
    total = len(x)
    num_nans = sum([1 for i in x if np.isnan(i)])
    return num_nans / float(total)

def name_to_int(ind):
    name = ind.__str__()
    hashed = int(hashlib.sha1(name).hexdigest(), 16)
    return np.float(hashed)

count_unique = lambda x: float(len(set(x)))

def trunc_func(decimals, func, x):
    return round(func(x), decimals)

def load_stats(names):
    get_stat_ind = lambda x: tools.Statistics(lambda ind: ind.fitness.values[x])
    stats = {i: get_stat_ind(ind) for ind, i in enumerate(names)}
    stats['inds'] = tools.Statistics(key=lambda ind: name_to_int(ind))
    stats = tools.MultiStatistics(**stats)
    stat_defs = (('avg', np.nanmean), 
                 ('std', np.nanstd), 
                 ('min', np.nanmin),
                 ('max', np.nanmax), 
                 ('nanr', nan_fit_ratio),
                 ('uniq', count_unique),)
    for i in stat_defs:
        stats.register(i[0], partial(trunc_func, 2, i[1]))
    return stats