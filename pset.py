import math
import operator
import random

import numpy as np
import pandas as pd
from functools import partial
from toolz.dicttoolz import keymap
from deap import gp


id = lambda x: x

trunc_rand_float = lambda x: round(random.uniform(0,x),3)

def xsrank(px):
    return px.rank(axis=1)

oppfuncs = (('sub', operator.sub),
            #('add', operator.add),
            #('mul', operator.mul),
            ('div', operator.div),
            ('min', np.minimum),
            ('max', np.maximum))

def shift(px, n):     
    n = int(math.floor(n))
    return pd.DataFrame(px).shift(n)

def change(px, n):
    return px / shift(px, n) - 1.

def rolling_apply(func, px, window):
    window = int(math.floor(window + 1))
    return px.rolling(window=window).apply(func=func)

rfuncs = (('rmean', np.nanmean),
          ('rstd', np.nanstd),
          ('rmax', np.nanmax),
          ('rmin', np.nanmin),)

rfuncs = list(map(lambda x: (x[0], partial(rolling_apply, x[1])), rfuncs))

def rolling_pairwise_corr(px1, px2, window):
    window = int(math.floor(window + 1))
    return px1.rolling(window=window).corr(other=px2)

def rolling_pairwise_cov(px1, px2, window):
    window = int(math.floor(window + 1))
    return px1.rolling(window=window).cov(other=px2)

rpfuncs = (('rpcorr', rolling_pairwise_corr),
           ('rpcov', rolling_pairwise_cov))

def load_pset(names):
    inp_dims = [pd.DataFrame for i in range(len(names))]
    pset = gp.PrimitiveSetTyped("MAIN", inp_dims, pd.DataFrame)

    for i in (operator.neg, abs, xsrank):
        pset.addPrimitive(i, [pd.DataFrame,], pd.DataFrame)

    for i in oppfuncs:
        pset.addPrimitive(i[1], [pd.DataFrame, pd.DataFrame], pd.DataFrame, name=i[0])
        # pset.addPrimitive(i[1], [pd.DataFrame, float], pd.DataFrame, name=i[0] + '_dfint')
        # pset.addPrimitive(i[1], [int, pd.DataFrame], pd.DataFrame, name=i[0] + '_intdf')
    
    for i in rfuncs:
        pset.addPrimitive(i[1], [pd.DataFrame, float], pd.DataFrame, name=i[0])  
    pset.addPrimitive(shift, [pd.DataFrame, float], pd.DataFrame, name='delay')
    pset.addPrimitive(change, [pd.DataFrame, float], pd.DataFrame, name='change')

    for i in rpfuncs:
        pset.addPrimitive(i[1], [pd.DataFrame, pd.DataFrame, float], pd.DataFrame, name=i[0])

    pset.addEphemeralConstant('rand30', partial(trunc_rand_float, 30), float)

    pset.addPrimitive(id, [float], float, name='id')
    pset.addPrimitive(id, [pd.DataFrame], pd.DataFrame, name='id2')

    args = dict(zip(range(len(names)), names))
    args = keymap(lambda x: 'ARG{0}'.format(x), args)
    pset.renameArguments(**args)
    return pset