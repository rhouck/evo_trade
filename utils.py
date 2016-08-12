import pandas as pd
import pygraphviz as pgv
from deap import gp
from IPython.display import Image

to_ret = lambda x: x.div(x.shift())

def get_fwd_ret(df, look_ahead_per):
    df_1 = df.applymap(lambda x: x + 1.)
    df_1 = df_1[::-1].rolling(window=look_ahead_per).apply(func=lambda x : x.prod())[::-1]
    return df_1.applymap(lambda x: x - 1.)

def draw_individual(ind):
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    fn = "data/graphs/tree.png"
    g.draw(fn)
    return fn

def display_individual(ind):
    fn = draw_individual(ind)
    return Image(filename=(fn))

def set_stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats

def calc_forecast_target_corr(forecast, target):
    return (pd.concat([forecast.stack(), target.stack()], axis=1)
            .dropna().corr().iloc[0,1])