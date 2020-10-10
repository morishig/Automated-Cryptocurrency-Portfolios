import math
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

time_window_days = 120
time_window_days = time_window_days - 1
rebalancing_period = 30

df = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/Prices.xlsx')

"""Get the dataframe and set indexes and columns"""
df['date'] = [datetime.strptime(d, '%d-%m-%Y') for d in df['date']]
dates = df['date']
df = df.set_index(df['date'])
del df['date']
cryptos = list(df.columns)

"""Initialize the variables for the loops"""
start_date = df.index[0]
start_date_plus1= start_date + timedelta(days=1)
start_portfolio = start_date + timedelta(days=time_window_days)
end_date=df.index[-1]

"""Calculate the logarithmic returns matrix"""
returns = [0] * (len(cryptos)) #first row of zeros
returns = pd.DataFrame([returns],columns=cryptos)

while start_date_plus1 <= end_date:
    daily_returns = list()

    for i in cryptos:
        t_minus1=df.loc[start_date_plus1 - timedelta(days=1)]
        t_minus1=t_minus1.loc[i]
        t=df.loc[start_date_plus1]
        t=t.loc[i]
        log_ret = math.log(t/t_minus1)
        daily_returns.append(log_ret)

    df_length = len(returns)
    returns.loc[df_length] = daily_returns
    start_date_plus1 = start_date_plus1 + timedelta(days=1)
returns = returns.set_index(dates)
returns=returns.replace([np.inf, -np.inf, np.nan],0)

"""Calculate weights and weighted returns"""
start_date_plus1 = start_date + timedelta(days=1)
start_date_i = start_date
start_portfolio_i = start_portfolio
first_return_day = start_portfolio + timedelta(days=1)
weights_index = 0

weighted_returns = [0] * (len(cryptos))
weighted_returns = pd.DataFrame([weighted_returns],columns=cryptos)

#Set to zero the weighted returns of the dates before the start of the portfolio
while start_date_plus1 <= start_portfolio:
    min_vol = [0] * (len(cryptos))
    df_length = len(weighted_returns)
    weighted_returns.loc[df_length] = min_vol
    start_date_plus1 = start_date_plus1 + timedelta(days=1)

#Set the first weights
rebalancing_day = start_portfolio

local_df = df[start_date_i: start_portfolio_i]
local_df = local_df.replace(0, np.nan).dropna(axis=1)
columns = list(local_df.columns)
drop = list(set(cryptos) - set(columns))

local_df = returns[start_date_i: start_portfolio_i]
local_df = local_df.iloc[1:]  # remove first row
local_df = local_df.drop(drop, axis=1)
columns = list(local_df.columns)

mu = expected_returns.mean_historical_return(local_df,returns_data=True,frequency=365)
rf = 0.00 #risk free rate - 0.02 (in this case = 0)
mu_excess = mu - rf

Sigma = risk_models.sample_cov(local_df,returns_data=True,frequency=365)

ones = [1] * (len(columns))
arr_ones = np.array(ones)
arr_ones = pd.DataFrame([arr_ones], columns=columns)

Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), Sigma.columns, Sigma.index)

x = arr_ones.dot(Sigma_inv)
y = x.dot(arr_ones.T)
y = y ** -1
y=y[0]
weights = Sigma_inv.multiply(y[0])
print(weights)
weights = weights.dot(arr_ones.T)
print(weights)

#Make the weights matrix sum = 1
#divisor = weights.sum()
#weights=weights.divide(divisor)

#weight_matrix = weights.to_frame()
weight_matrix = weights.T
weight_matrix['date'] = rebalancing_day
weight_matrix = weight_matrix.set_index('date')

k = weight_matrix.abs()
k = k.sum(axis=1)
weight_matrix = weight_matrix.divide(k, axis=0)

print(weight_matrix)

start_portfolio_i = start_portfolio_i + timedelta(days=1)
start_date_i = start_date_i + timedelta(days=1)

while start_portfolio_i <= end_date:
    min_vol = list()
    rebalancing_day_check=start_portfolio_i-start_portfolio
    rebalancing_day_check=rebalancing_day_check.days

    #rebalancing and returns calculation
    if rebalancing_day_check % rebalancing_period == 0 :

        for i in cryptos:
            if i not in columns: #All the cryptos not in the weight names set to 0
                min_vol.append(0)

            else: #Returns calculation
                r = returns.loc[start_portfolio_i]
                r = r.loc[i]
                weight_minus1 = weight_matrix.loc[rebalancing_day]
                weight_minus1 = weight_minus1.loc[i]
                c = r * weight_minus1
                min_vol.append(c)

        local_df = df[start_date_i: start_portfolio_i]
        local_df = local_df.replace(0, np.nan).dropna(axis=1)
        columns = list(local_df.columns)
        drop = list(set(cryptos) - set(columns))

        local_df = returns[start_date_i: start_portfolio_i]
        local_df = local_df.drop(drop, axis=1)
        columns = list(local_df.columns)

        rebalancing_day = rebalancing_day + timedelta(days=rebalancing_period)

        mu = expected_returns.mean_historical_return(local_df,returns_data=True,frequency=365)
        mu_excess = mu - rf

        Sigma = risk_models.sample_cov(local_df, returns_data=True, frequency=365)
        ones = [1] * (len(columns))
        arr_ones = np.array(ones)
        arr_ones = pd.DataFrame([arr_ones], columns=columns)

        Sigma_inv = pd.DataFrame(np.linalg.pinv(Sigma.values), Sigma.columns, Sigma.index)

        x = arr_ones.dot(Sigma_inv)
        y = x.dot(arr_ones.T)
        y = y ** -1
        y = y[0]
        weights = Sigma_inv.multiply(y[0])
        weights = weights.dot(arr_ones.T)

        # Make the weights matrix sum = 1
        #divisor = weights.sum()
        #weights = weights.divide(divisor)

        #weights = weights.to_frame()
        weights = weights.T
        weights['date'] = rebalancing_day
        weights = weights.set_index('date')

        #scaling factor k, making the sum of all short-long positions = 1
        k = weights.abs()
        k = k.sum(axis=1)
        weights = weights.divide(k, axis=0)

        weight_matrix = [weight_matrix, weights]
        weight_matrix = pd.concat(weight_matrix, ignore_index=False)

        df_length = len(weighted_returns)
        weighted_returns.loc[df_length] = min_vol

    # Returns calculation
    else:
        for i in cryptos:
            if i not in columns:
                min_vol.append(0)
            else:
                r = returns.loc[start_portfolio_i]
                r = r.loc[i]
                weight_minus1 = weight_matrix.loc[rebalancing_day]
                weight_minus1 = weight_minus1.loc[i]
                c = r * weight_minus1
                min_vol.append(c)

        df_length = len(weighted_returns)
        weighted_returns.loc[df_length] = min_vol

    start_portfolio_i = start_portfolio_i + timedelta(days=1)
    start_date_i = start_date_i + timedelta(days=1)
weighted_returns = weighted_returns.set_index(dates)
weighted_returns['portfolio return'] = weighted_returns.sum(axis=1)

"""Calculation of the portfolio value"""
initial_value = 1
start_portfolio_i = start_portfolio + timedelta(days=1)

portfolio_value = [0] * (time_window_days)
portfolio_value.append(initial_value)
portfolio_return_index = time_window_days


while start_portfolio_i <= end_date:

    ret = weighted_returns.loc[start_portfolio_i]
    ret = ret.loc['portfolio return']

    if ret == 0:
        portfolio_value_previous = portfolio_value[portfolio_return_index]
        current_value = portfolio_value_previous
        portfolio_value.append(current_value)

        start_portfolio_i = start_portfolio_i + timedelta(days=1)
        portfolio_return_index = portfolio_return_index + 1
    else:
        portfolio_value_previous = portfolio_value[portfolio_return_index]
        current_value = portfolio_value_previous * math.exp(ret)
        portfolio_value.append(current_value)

        start_portfolio_i = start_portfolio_i + timedelta(days=1)
        portfolio_return_index = portfolio_return_index + 1

weighted_returns['portfolio_value'] = portfolio_value

#Save the dataframes to Excel file
weighted_returns.to_excel(r'/Users/gmmtakane/Desktop/Thesis/weighted_returns mv monthly' + '.xlsx', index = True)
weight_matrix.to_excel(r'/Users/gmmtakane/Desktop/Thesis/weights mv monthly' + '.xlsx')
returns.to_excel(r'/Users/gmmtakane/Desktop/Thesis/log returns mv monthly' + '.xlsx')
