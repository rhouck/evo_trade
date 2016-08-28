import math
import operator
import random

import numpy as np
import pandas as pd
from functools import partial
from toolz.dicttoolz import keymap
from deap import gp
from scipy.stats import norm


def idf(x):
    return x

def trunc_rand_float(x): 
    return round(random.uniform(0, x), 3)

# core funcs
def xsrank(px):
    return px.rank(axis=1)

oppfuncs = (('sub', operator.sub),
            #('add', operator.add),
            ('mul', operator.mul),
            ('div', operator.div),
            ('min', np.minimum),
            ('max', np.maximum))

def shift(px, n):     
    n = int(math.floor(n))
    return pd.DataFrame(px).shift(n)

def change(px, n):
    return px / shift(px, n) - 1.

# rolling funcs
def rolling_apply(func, px, window):
    window = int(math.floor(window + 1))
    return px.rolling(window=window).apply(func=func)

def tsrank(data): 
    return data.argsort()[-1] + 1

rfuncs = (('rmean', np.nanmean),
          ('rstd', np.nanstd),
          ('rmax', np.nanmax),
          ('rmin', np.nanmin),
          ('rtsrank', tsrank),)

rfuncs = dict(map(lambda x: (x[0], partial(rolling_apply, x[1])), rfuncs))

def get_exp_window(px, halflife):
    return px.ewm(halflife=halflife, ignore_na=True, 
                  min_periods=int(halflife*.75))

def ermean(px, halflife):
    return get_exp_window(px, halflife).mean()

def erstd(px, halflife):
    return get_exp_window(px, halflife).std()

def rolling_pairwise_corr(px1, px2, window):
    window = int(math.floor(window + 1))
    return px1.rolling(window=window).corr(other=px2)

def rolling_pairwise_cov(px1, px2, window):
    window = int(math.floor(window + 1))
    return px1.rolling(window=window).cov(other=px2)

rpfuncs = dict((('rpcorr', rolling_pairwise_corr),
                ('rpcov', rolling_pairwise_cov)))

# complex funcs
def change_smooth(px, n, window):
    px_d = change(px, n)
    return rolling_apply(np.nanmean, px_d, window)

def vol_scl(px, window):
    px_std = rolling_apply(np.nanstd, px, window)
    return px.div(px_std)

def ew_vol_scl(px, halflife):
    px_std = get_exp_window(px, halflife).std()
    return px.div(px_std)

def xsrank_corr(px1, px2, window):
    rpx1 = xsrank(px1)
    rpx2 = xsrank(px2)
    return rolling_pairwise_corr(rpx1, rpx2, window)

def tsrank_corr(px1, px2, r_window, c_window):
    rpx1 = rfuncs['rtsrank'](px1, r_window)
    rpx2 = rfuncs['rtsrank'](px2, r_window)
    return rolling_pairwise_corr(rpx1, rpx2, c_window)

def rank_as_pct(rank, size):
    rank /= size
    max_val = (size - 1.) / size
    return rank.replace(1, max_val)

def xsrank_norm(px):
    rank = xsrank(px)
    rank = rank_as_pct(rank, rank.shape[1])
    return rank.apply(norm.ppf)

def tsrank_norm(px, window):
    rank = rfuncs['rtsrank'](px, window)
    rank = rank_as_pct(rank, window + 1)
    return rank.apply(norm.ppf)

def load_pset(names):
    inp_dims = [pd.DataFrame for i in range(len(names))]
    pset = gp.PrimitiveSetTyped("MAIN", inp_dims, pd.DataFrame)

    # core funcs
    for i in (operator.neg, abs, xsrank):
        pset.addPrimitive(i, [pd.DataFrame,], pd.DataFrame)

    for i in oppfuncs:
        pset.addPrimitive(i[1], [pd.DataFrame, pd.DataFrame], pd.DataFrame, name=i[0])
        # pset.addPrimitive(i[1], [pd.DataFrame, float], pd.DataFrame, name=i[0] + '_dfflt')
        # pset.addPrimitive(i[1], [float, pd.DataFrame], pd.DataFrame, name=i[0] + '_fltdf')
    
    pset.addPrimitive(shift, [pd.DataFrame, float], pd.DataFrame, name='delay')
    pset.addPrimitive(change, [pd.DataFrame, float], pd.DataFrame, name='change')


    # rolling funcs
    for i in rfuncs.items():
        pset.addPrimitive(i[1], [pd.DataFrame, float], pd.DataFrame, name=i[0])

    for i in (ermean, erstd):
        pset.addPrimitive(i, [pd.DataFrame, float], pd.DataFrame)

    for i in rpfuncs.items():
        pset.addPrimitive(i[1], [pd.DataFrame, pd.DataFrame, float], pd.DataFrame, name=i[0])
    

    # complex funcs
    pset.addPrimitive(change_smooth, [pd.DataFrame, float, float], pd.DataFrame, name='ch_sm')  
    for i in (vol_scl, ew_vol_scl):
        pset.addPrimitive(i, [pd.DataFrame, float], pd.DataFrame)
    pset.addPrimitive(xsrank_corr, [pd.DataFrame, pd.DataFrame, float], pd.DataFrame)
    pset.addPrimitive(tsrank_corr, [pd.DataFrame, pd.DataFrame, float, float], pd.DataFrame)
    pset.addPrimitive(xsrank_norm, [pd.DataFrame,], pd.DataFrame)
    pset.addPrimitive(tsrank_norm, [pd.DataFrame, float], pd.DataFrame)
    
    pset.addEphemeralConstant('rand60', partial(trunc_rand_float, 60), float)

    dtypes = enumerate((float, pd.DataFrame))
    named_dtypes = map(lambda x: ('id' + str(x[0]), x[1]), dtypes)
    for i in named_dtypes:
        pset.addPrimitive(idf, [i[1]], i[1], name=i[0])

    args = dict(zip(range(len(names)), names))
    args = keymap(lambda x: 'ARG{0}'.format(x), args)
    pset.renameArguments(**args)
    return pset