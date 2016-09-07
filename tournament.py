import random
from itertools import chain

import numpy as np
import pandas as pd
import dill
from deap import algorithms

from utils import drop_id_funcs


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


import hashlib
def name_to_int(ind):
    name = ind.__str__()
    hashed = int(hashlib.sha1(name).hexdigest(), 16)
    return np.float(hashed)
count_unique = lambda x: float(len(set(x)))



def run_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                   checkpoint_fn, start_gen, pset, randstate=get_seed_state(), cp_freq=5, seed_scale=3):
    
    random.setstate(randstate)
    start_len = len(pop) / seed_scale
    for gen in range(start_gen, start_gen + ngen):
        
        if gen:
            pop_start = [ind for ind in pop]
            #n = count_unique([name_to_int(i) for i in pop_start])
            #print('pop_start:\t{0}\t{1}'.format(len(pop_start), n))


        #pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        #pop = [drop_id_funcs(ind, toolbox, pset) for ind  in pop]

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # filt invalid and upsample to replace invalid inds
        pop = [ind for ind in pop if not is_nan(ind)]
        #pop = upsample_pop(pop, start_len)
        
        #n = count_unique([name_to_int(i) for i in pop])
        #print('new pop:\t{0}\t{1}'.format(len(pop), n))

        if gen:
            pop.extend(pop_start)
            #n = count_unique([name_to_int(i) for i in pop])
            #print('pop combo:\t{0}\t{1}'.format(len(pop), n))

        # drop duplicates
        pop = list(pd.DataFrame([(i, i.__str__()) for i in pop]).drop_duplicates(1)[0])
        
        pop = toolbox.select(pop, k=start_len)

        #n = count_unique([name_to_int(i) for i in pop])
        #print('sel pop:\t{0}\t{1}'.format(len(pop), n))

        # log stats
        record = stats.compile(pop)
        hof.update(pop)
        log.record(gen=gen, evals=len(invalid_ind), **record)
        print(log.stream)

        if (gen + 1) % cp_freq == 0:
            checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, 
                                  random.getstate())
  
    return pop, hof, log