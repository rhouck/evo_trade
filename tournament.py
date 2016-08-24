import random

import numpy as np
import dill
from deap import algorithms


def get_seed_state():
    random.seed(0)
    return random.getstate()

is_nan = lambda ind: any(map(np.isnan, ind.fitness.values))

def run_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                   checkpoint_fn, start_gen, randstate=get_seed_state()):
    
    random.setstate(randstate)
    start_len = len(pop)
    for gen in range(start_gen, start_gen + ngen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

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
            cp = dict(pop=pop, gen=gen, hof=hof, log=log, 
                      randstate=random.getstate())

            with open(checkpoint_fn, 'wb') as f:
                f.write(dill.dumps(cp, 2))

    return pop, hof, log
