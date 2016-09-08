import random
import copy
from itertools import chain

import numpy as np
import pandas as pd
import dill
from deap import algorithms
from toolz import compose

#from utils import aggregate_attr


def get_seed_state():
    random.seed(0)
    return random.getstate()

is_nan = lambda ind: any(map(np.isnan, ind.fitness.values))

def upsample_pop(pop, size):
    if len(pop) > size:
        raise Exception('`pop` already larger than `size`')

    full = size / len(pop)
    full_pops = [pop for i in range(full)]
    full_pops = list(chain(*full_pops))
    
    n_extra = size % len(pop)
    extra = list(np.random.choice(pop, size=n_extra, replace=False))
    
    return full_pops + extra

def checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, randstate):
    cp = dict(pop=pop, gen=gen, hof=hof, log=log, randstate=randstate)
    with open(checkpoint_fn, 'wb') as f:
        f.write(dill.dumps(cp, 2))

def load_checkpoint(checkpoint_fn):
    cp = dill.load(open(checkpoint_fn, 'r'))
    items = ('pop', 'gen', 'hof', 'log', 'randstate')
    return map(lambda x: cp[x], items) 

def run_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                   checkpoint_fn, start_gen, pset, cp_freq, seed_scale, 
                   randstate=get_seed_state(), format_new_ind=lambda x: x):
    
    random.setstate(randstate)
    start_len = len(pop) / seed_scale
    for gen in range(start_gen, start_gen + ngen):
                    
        if gen:
            pop_start =  copy.deepcopy(pop)
            pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
            pop.extend(pop_start)

        nevals = sum([1 for ind in pop if not ind.fitness.valid])
        pop = [compose(toolbox.evaluate, format_new_ind)(ind)
                    if not ind.fitness.valid else ind for ind in pop]
        
        # drop invalid and duplicate inds
        pop = [ind for ind in pop if not is_nan(ind)]
        pop = list(pd.DataFrame([(i, i.__str__()) for i in pop]).drop_duplicates(1)[0])

        pop = toolbox.select(pop, k=start_len)

        # log stats
        record = stats.compile(pop)
        hof.update(pop)
        log.record(gen=gen, evals=nevals, **record)
        print(log.stream)

        if (gen + 1) % cp_freq == 0:
            checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, 
                                  random.getstate())
  
    return pop, hof, log

# def run_add_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
#                       checkpoint_fn, start_gen, pset, cp_freq, 
#                       randstate=get_seed_state(), format_new_ind=lambda x: x):

#     random.setstate(randstate)
#     start_len = len(pop)
#     for gen in range(start_gen, start_gen + ngen):
        
        
#         pop_start = [ind for ind in pop]
#         agg_pop_start = {i: aggregate_attr(pop, i) for i in ( 'scores','holdings')}
#         agg_pop_start['len'] = len(pop)
#         for i in pop_start:
#             setattr(i, 'fitness', toolbox.fitness())

#         pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
#         pop = [format_new_ind(ind) for ind in pop]
#         pop = [drop_id_funcs(toolbox, pset, ind) for ind  in pop]

#         pop.extend(pop_start)

#         # evaluate the individuals with an invalid fitness
#         invalid_ind = [ind for ind in pop if not ind.fitness.valid]
#         for ind in invalid_ind:
#             toolbox.evaluate(ind, agg_pop)
#         # agg_pops = [agg_pop_start for i in range(len(invalid_ind))]
#         # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind, agg_pops)
#         # for ind, fit in zip(invalid_ind, fitnesses):
#         #     ind.fitness.values = fit['objectives']
#         #     for i in ('scores', 'holdings'):
#         #         setattr(ind, i, fit[i])

#         # filt invalid and upsample to replace invalid inds
#         pop = [ind for ind in pop if not is_nan(ind)]
#         #pop = upsample_pop(pop, start_len)
        
#         # drop duplicates
#         pop = list(pd.DataFrame([(i, i.__str__()) for i in pop]).drop_duplicates(1)[0])
        
#         pop = toolbox.select(pop, k=start_len)

#         # log stats
#         record = stats.compile(pop)
#         hof.update(pop)
#         log.record(gen=gen, evals=len(invalid_ind), **record)
#         print(log.stream)

#         if (gen + 1) % cp_freq == 0:
#             checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, 
#                                   random.getstate())
  
#     return pop, hof, log