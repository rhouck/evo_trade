import numpy as np

from functools import partial
from deap import tools

def nan_fit_ratio(x):
    total = len(x)
    num_nans = sum([1 for i in x if np.isnan(i)])
    return num_nans / float(total)

def trunc_func(decimals, func, x):
    return round(func(x), decimals)

def load_stats(names):
    get_stat_ind = lambda x: tools.Statistics(lambda ind: ind.fitness.values[x])
    stats = {i: get_stat_ind(ind) for ind, i in enumerate(names)}
    stats = tools.MultiStatistics(**stats)
    stat_defs = (('avg', np.nanmean), 
                 ('std', np.nanstd), 
                 ('min', np.nanmin),
                 ('max', np.nanmax), 
                 ('nanr', nan_fit_ratio),)
    for i in stat_defs:
        stats.register(i[0], partial(trunc_func, 2, i[1]))
    return stats