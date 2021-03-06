import math
import re
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats
from toolz import pipe, compose
from toolz.curried import map as cmap
from deap import gp

try:
    import pygraphviz as pgv
    from IPython.display import Image

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
except:
    pass
    
def drop_id_funcs_str(s):
    p = re.compile("id(.?)\(")
    ps = p.search(s)
    if ps:
        p = ps.group(0)
        ss = s.split(p, 1)
        a = ss[0]
        b = ss[1].replace(')', '', 1)
        s = a + b
        return drop_id_funcs_str(s)
    return s

def drop_id_funcs(toolbox, pset, ind):
    s = ind.__str__() 
    s_filt = drop_id_funcs_str(s)
    if s == s_filt:
        return ind
    ind_filt = gp.PrimitiveTree.from_string(s_filt, pset)
    ind_filt.fitness = toolbox.fitness()
    if ind.fitness.valid:  
        ind_filt.fitness.values = ind.fitness.values
    return ind_filt

first = lambda x: x[0]
second = lambda x: x[1]
third = lambda x: x[2]

to_ret = lambda x: x.div(x.shift())
calc_port_returns = lambda hld, ret: hld.mul(ret, axis=1).sum(axis=1)
calc_ir = lambda x: (x.mean() * 252.) / (x.std() * math.sqrt(252.))
xscore = lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

def scale_returns(returns, risk=.05):
    risk_daily = risk / math.sqrt(252.)
    scaler = (risk_daily / returns.std())
    return returns * scaler

def get_fwd_ret(df, look_ahead_per):
    df_1 = df.applymap(lambda x: x + 1.)
    df_1 = df_1[::-1].rolling(window=look_ahead_per).apply(func=lambda x : x.prod())[::-1]
    return df_1.applymap(lambda x: x - 1.)

def calc_dfs_corr(a, b):
    return (pd.concat([a.stack(), b.stack()], axis=1)
            .dropna().corr().iloc[0,1])

def scores_to_holdings(rm, scores):
    scores_xs = xscore(scores)
    alphas = scores_xs.mul(rm.vol) * .05
    return rm.calc_holdings(alphas)
 
def calc_nan_rows_ratio(df):
    clean = df.dropna(how='all').shape[0]
    full = df.shape[0]
    return 1. - (clean / float(full))

def calc_two_sided_rel_to(holdings):
    delta = holdings.sub(holdings.shift()).abs().sum(axis=1) / 2.
    lev = holdings.abs().sum(axis=1)
    return delta / lev

calc_ann_to = lambda hld: calc_two_sided_rel_to(hld).mean() * 252

def calc_tilt(px):
    scale = px.stack().std()
    px_scaled = px / scale
    return px_scaled.mean().abs().mean()

def apply_xscore_scipy(func, scores):
    data = xscore(scores).stack().values
    return float(abs(func(data)))

get_scores_skew = partial(apply_xscore_scipy, scipy.stats.mstats.skew)
get_scores_kurt = partial(apply_xscore_scipy, scipy.stats.mstats.kurtosis)

def get_forecast(toolbox, individual, pxs):
    func = toolbox.compile(expr=individual)
    return func(*pxs)

def get_ind_counts(pop):
    df = pd.DataFrame([(i, i.__str__()) for i in pop])
    df.columns = ['ind', 'str']
    counts = df.groupby('str').count()['ind'].to_dict()
    df['counts'] = df['str'].map(lambda x: counts[x])
    return list(df.drop_duplicates('str')[['ind', 'counts']].values)

def aggregate_attr(pop, attr, transform=lambda x: x):
    get_vals = lambda x: getattr(x, attr)
    stack = lambda x: x.stack()
    func = compose(stack, transform, get_vals)
    return (pd.concat([func(i) for i in pop], axis=1)
            .mean(axis=1).unstack())

class RiskModel(object):
    
    def __init__(self, returns, halflife=252):
        if returns.shape[0] != returns.dropna(how='all').shape[0]:
            raise Exception('returns cannot contain nan rows')
        
        self.returns = returns
        min_per = int(halflife * .75)
        returns_grouped = returns.ewm(halflife=halflife,ignore_na=True,min_periods=min_per)
        self.vol = returns_grouped.std().dropna(how='all') 
        self.cov = returns_grouped.cov().dropna(how='all', axis='items')
        self.inv_cov = self.invert_cov(self.cov)    
    
    def invert_cov(self, cov):
        invs = {}
        for date in cov.items:
            cov_full = self.cov[date].dropna(how='all').dropna(how='all', axis=1)
            invs[date] = pd.DataFrame(np.linalg.inv(cov_full), 
                                      index=cov_full.index, 
                                      columns=cov_full.columns)
        return pd.Panel(invs)

    def calc_holdings(self, alphas):
        dates = self.inv_cov.items
        holdings = {}
        for date in dates:
            inv_cov_full = self.inv_cov.loc[date].dropna(how='all').dropna(how='all', axis=1)
            alphas_full = alphas.loc[date, inv_cov_full.columns].dropna()
            inv_cov_full = inv_cov_full.loc[alphas_full.index, alphas_full.index]
            holdings[date] = inv_cov_full.dot(alphas_full)
        return pd.DataFrame(holdings).T