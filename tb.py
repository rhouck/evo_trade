import operator

from deap import base
from deap import creator
from deap import gp
from deap import tools

def load_toolbox(weights, pset, pxs, eval_func, tournsize):
    creator.create("FitnessMax", base.Fitness, weights=weights)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", eval_func, pxs=pxs)
    #toolbox.decorate("evaluate", tools.DeltaPenality(is_feasible, 7.0, distance))
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
    return toolbox