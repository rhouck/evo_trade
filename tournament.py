import random
import copy
from itertools import chain
from functools import partial

import numpy as np
import pandas as pd
import dill
from deap import algorithms
from toolz import compose


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

def run_kbest_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, checkpoint_fn, 
                         start_gen, pset, cp_freq, seed_scale, randstate, opt_func):
    
    random.setstate(randstate)
    start_len = len(pop) / seed_scale
    for gen in range(start_gen, start_gen + ngen):
        
        pop, nevals = opt_func(pop, gen)

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

get_nevals = lambda pop: sum([1 for ind in pop if not ind.fitness.valid])

def optimize_inds(toolbox, cxpb, mutpb, format_new_ind, pop, gen):
    if gen:
        pop_start =  copy.deepcopy(pop)
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        pop.extend(pop_start)

    nevals = get_nevals(pop)
    pop = [compose(toolbox.evaluate, format_new_ind)(ind)
                if not ind.fitness.valid else ind for ind in pop]
    return pop, nevals

def run_std_tourn(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                  checkpoint_fn, start_gen, pset, cp_freq, seed_scale, 
                  randstate=get_seed_state(), format_new_ind=lambda x: x):
    
    opt_func = partial(optimize_inds, toolbox, cxpb, mutpb, format_new_ind)
    return run_kbest_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                                checkpoint_fn, start_gen, pset, cp_freq, seed_scale, 
                                randstate, opt_func)


def optimize_agg(toolbox, cxpb, mutpb, format_new_ind, agg_func, pop, gen):
    pop_start =  copy.deepcopy(pop)
    agg_stats = agg_func(pop_start)
    for i in pop_start:
        setattr(i, 'fitness', toolbox.fitness())
    
    pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
    pop = [format_new_ind(ind) for ind in pop]
    pop.extend(pop_start)
    
    nevals = get_nevals(pop)
    pop = [toolbox.evaluate(ind, agg_stats) for ind in pop]

    return pop, nevals

def run_add_tourn(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                  checkpoint_fn, start_gen, pset, cp_freq, agg_func,
                  randstate=get_seed_state(), format_new_ind=lambda x: x):
    
    opt_func = partial(optimize_agg, toolbox, cxpb, mutpb, format_new_ind, agg_func)
    return run_kbest_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                                checkpoint_fn, start_gen, pset, cp_freq, 1, 
                                randstate, opt_func)