import operator

import numpy as np
from deap import base
from deap import creator
from deap import gp
from deap import tools


def load_toolbox(weights, pset, pxs, eval_func, tournsize, height_range=(2,4)):
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=height_range[0], max_=height_range[1])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", eval_func, pxs=pxs)
    #infeas_vals = (np.nan for i in range(len(weights)))
    #toolbox.decorate("evaluate", tools.DeltaPenality(feasibility_func, infeas_vals))
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=height_range[1])
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    #toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=height_range[1]))
    #toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=height_range[1]))
    return toolbox