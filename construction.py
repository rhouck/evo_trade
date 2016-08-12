import numpy as np
import pandas as pd
from functools import partial


def shift(px, n):     
    return pd.DataFrame(px).shift(n)
#     e = np.empty_like(px)
#     if n == 0:
#         return px
#     elif n >= 0:         
#         e[:n] = np.nan         
#         e[n:] = px[:-n]     
#     else:         
#         e[n:] = np.nan       
#         e[:n] = px[-n:]     
#     return e

# def rolling_window(a, window):
#     a = a.T
#     shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#     strides = a.strides + (a.strides[-1],)
#     return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# def add_nan_rows(px, window):
#     nr = np.empty((window,px.shape[1],))
#     nr[:] = np.nan
#     return np.concatenate([nr, px], axis=0)

# def transform(px, func, window, delay):
#     px_sub = rolling_window(px, window + 1)
#     px_sub = func(px_sub, axis=2).T
#     px_sub = add_nan_rows(px_sub, window)
#     px_sub = shift(px_sub, delay)
#     return px_sub

def rolling_apply(func, px, window):
    #return pd.rolling_apply(px, window=window, func=func, min_periods=0)
    return px.rolling(window=window).apply(func=func)
#     pxt = rolling_window(px, window + 1)
#     pxt = func(pxt, axis=2).T
#     pxt = add_nan_rows(pxt, window)
#     return pxt

rfuncs = (('rmean', np.nanmean),
          ('rstd', np.nanstd),
          ('rmax', np.nanmax),
          ('rmin', np.nanmin),)

rfuncs = list(map(lambda x: (x[0], partial(rolling_apply, x[1])), rfuncs))

id = lambda x: x

class RiskModel(object):

    
    def __init__(self, returns, halflife=6*22):
        if returns.shape[0] != returns.dropna(how='all').shape[0]:
            raise Exception('returns cannot contain nan rows')
        
        self.returns = returns
        returns_grouped = returns.ewm(halflife=halflife,ignore_na=True,min_periods=0)
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

    def calc_holdings(alphas, risk_scale=.05):
        pass



