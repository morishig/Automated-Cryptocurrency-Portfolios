from pypfopt import CLA, plotting
import mlfinlab as mf
import math
import pandas as pd
import numpy as np
import pyRMT as rmt
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from mlfinlab.portfolio_optimization.mean_variance import MeanVarianceOptimisation
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity

time_window_days = 120
time_window_days_minus1 = time_window_days - 1
rebalancing_period = 7

df = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/Prices.xlsx')

"""Get the dataframe and set indexes and columns"""
df['date'] = [datetime.strptime(d, '%d-%m-%Y') for d in df['date']]
dates = df['date']
df = df.set_index(df['date'])
del df['date']
cryptos = list(df.columns)

"""Initialize the variables for the loops"""
start_date = df.index[0]
start_date_plus1 = start_date + timedelta(days=1)
start_portfolio = start_date + timedelta(days=time_window_days_minus1)
end_date = df.index[-1]

"""Calculate the logarithmic returns matrix"""
returns = [0] * (len(cryptos))  # first row of zeros
returns = pd.DataFrame([returns], columns=cryptos)

while start_date_plus1 <= end_date:
    daily_returns = list()

    for i in cryptos:
        t_minus1 = df.loc[start_date_plus1 - timedelta(days=1)]
        t_minus1 = t_minus1.loc[i]
        t = df.loc[start_date_plus1]
        t = t.loc[i]
        log_ret = math.log(t / t_minus1)
        daily_returns.append(log_ret)

    df_length = len(returns)
    returns.loc[df_length] = daily_returns
    start_date_plus1 = start_date_plus1 + timedelta(days=1)
returns = returns.set_index(dates)
returns = returns.replace([np.inf, -np.inf, np.nan], 0)

"""Calculate weights and weighted returns"""
start_date_plus1 = start_date + timedelta(days=1)
start_date_i = start_date
start_portfolio_i = start_portfolio
first_return_day = start_portfolio + timedelta(days=1)
weights_index = 0

weighted_returns = [0] * (len(cryptos))
weighted_returns = pd.DataFrame([weighted_returns], columns=cryptos)

# Set to zero the weighted returns of the dates before the start of the portfolio
while start_date_plus1 <= start_portfolio:
    min_vol = [0] * (len(cryptos))
    df_length = len(weighted_returns)
    weighted_returns.loc[df_length] = min_vol
    start_date_plus1 = start_date_plus1 + timedelta(days=1)

# Set the first weights
rebalancing_day = start_portfolio

local_df = df[start_date_i: start_portfolio_i]
local_df = local_df.replace(0, np.nan).dropna(axis=1)
columns = list(local_df.columns)
drop = list(set(cryptos) - set(columns))

local_df = returns[start_date_i: start_portfolio_i]
local_df = local_df.iloc[1:]  # remove first row
local_df = local_df.drop(drop, axis=1)
columns = list(local_df.columns)

mu = expected_returns.mean_historical_return(local_df, returns_data=True, frequency=365)
rf = 0.00  # risk free rate - 0.02 (in this case = 0)
mu_excess = mu - rf

Sigma = risk_models.sample_cov(local_df, returns_data=True, frequency=365)
Sigma = pd.DataFrame(data=Sigma, index=columns, columns=columns)

ones = [1] * (len(columns))
arr_ones = np.array(ones)
arr_ones = pd.DataFrame([arr_ones], columns=columns)

Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), Sigma.columns, Sigma.index)

x = arr_ones.dot(Sigma_inv)
y = x.dot(mu_excess)
y = y ** -1
weights = Sigma_inv.multiply(float(y))
weights = weights.dot(mu_excess)

weight_matrix = weights.to_frame()
weight_matrix = weight_matrix.T
weight_matrix['date'] = rebalancing_day
weight_matrix = weight_matrix.set_index('date')

#k = weight_matrix.abs()
#k = k.sum(axis=1)
#weight_matrix = weight_matrix.divide(k, axis=0)

"""print(weight_matrix)
cla = CLA(mu, Sigma)
cla.max_sharpe()
#cla.min_volatility()
cla.portfolio_performance(verbose=True);
ax = plotting.plot_efficient_frontier(cla, points=100,show_assets=True, showfig=True)"""


"""#Plot Efficient Frontier
mvo = MeanVarianceOptimisation()
plot = mvo.plot_efficient_frontier(covariance=Sigma,expected_asset_returns=mu_excess,risk_free_rate=0.0, max_return=12.0)
"""

"""#plot Covariance
plotting.plot_covariance(Sigma) 
"""

"""#Plots Covariance Matrix after the Eigenvalue Clipping
RMT = rmt.clipped(Sigma,return_covariance=True)
RMT = pd.DataFrame(data=RMT,index=columns,columns=columns)
plotting.plot_covariance(RMT)
"""

"""#Plot HRP
hrp = HierarchicalRiskParity()
hrp.allocate(asset_returns=local_df)
plt.figure(figsize=(10,8))
hrp.plot_clusters(columns)
plt.title('Hierarchical Risk Parity Dendrogram', size=18)
plt.xticks(rotation=45)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()
"""


cc = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/Portfolios.xlsx')
print(cc)
lst = list(cc.columns)
print(lst)

cc = cc.set_index(pd.to_datetime(cc['date']), drop=True)
del cc['date']
print (cc)

cc.plot()
plt.title('Portfolio Performance')
plt.ylabel('P&L')
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()
