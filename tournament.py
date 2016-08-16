import random

import dill
from deap import algorithms


def get_seed_state():
    random.seed(0)
    return random.getstate()

def run_tournament(pop, toolbox, cxpb, mutpb, ngen, stats, hof, log, 
                   checkpoint_fn, start_gen, randstate=get_seed_state()):
    
    random.setstate(randstate)

    for gen in range(start_gen, start_gen + ngen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        record = stats.compile(pop)
        log.record(gen=gen, evals=len(invalid_ind), **record)
        pop = toolbox.select(pop, k=len(pop))
        print(log.stream)  

        if gen % 10 == 0:
            cp = dict(pop=pop, gen=gen, hof=hof, log=log, 
                      randstate=random.getstate())

            with open(checkpoint_fn, 'wb') as f:
                f.write(dill.dumps(cp, 2))

    return pop, hof, log