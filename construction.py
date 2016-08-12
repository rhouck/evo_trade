import numpy as np
import pandas as pd
from functools import partial


def shift(px, n):     
    return pd.DataFrame(px).shift(n)

def rolling_apply(func, px, window):
    return px.rolling(window=window).apply(func=func)

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



