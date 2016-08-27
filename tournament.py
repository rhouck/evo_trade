import random

import numpy as np
import dill
from deap import algorithms

from utils import drop_id_funcs


def get_seed_state():
    random.seed(0)
    return random.getstate()

is_nan = lambda ind: any(map(np.isnan, ind.fitness.values))

def checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, randstate):
    cp = dict(pop=pop, gen=gen, hof=hof, log=log, randstate=randstate)
    with open(checkpoint_fn, 'wb') as f:
        f.write(dill.dumps(cp, 2))

def load_checkpoint(checkpoint_fn):
    cp = dill.load(open(checkpoint_fn, 'r'))
    items = ('pop', 'gen', 'hof', 'log', 'randstate')
    return map(lambda x: cp[x], items) 

def run_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                   checkpoint_fn, start_gen, pset, randstate=get_seed_state()):
    
    random.setstate(randstate)
    start_len = len(pop)
    for gen in range(start_gen, start_gen + ngen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        pop = [drop_id_funcs(ind, toolbox, pset) for ind  in pop]

        # evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # log stats
        record = stats.compile(pop)
        log.record(gen=gen, evals=len(invalid_ind), **record)
        print(log.stream)

        # filt invalid and select best inds
        pop = [ind for ind in pop if not is_nan(ind)]
        hof.update(pop)
        pop = toolbox.select(pop, k=start_len)
        
        if gen % 10 == 0:
            checkpoint_tournament(checkpoint_fn, pop, gen, hof, log, 
                                  random.getstate())
  
    return pop, hof, log