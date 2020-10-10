import math
import pandas as pd
import numpy as np
from scipy import stats as st
from datetime import timedelta
from datetime import datetime
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import statsmodels as sm
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from mlfinlab.portfolio_optimization.hrp import HierarchicalRiskParity

rf=0
time_window_days = 120
time_window_days_minus1 = time_window_days - 1
rebalancing_period = 7

df = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/CRIX.xlsx')

"""Get the dataframe and set indexes and columns"""
df['date'] = [datetime.strptime(d, '%d-%m-%Y') for d in df['date']]
dates = df['date']
df = df.set_index(df['date'])
del df['date']
cryptos = list(df.columns)

"""Initialize the variables for the loops"""
start_date = df.index[0]
start_date_plus1= start_date + timedelta(days=1)
start_portfolio = start_date + timedelta(days=time_window_days_minus1)
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

"""Calculate the fully invested portfolios in each cryptocurrency"""
for i in cryptos:

    initial_value = 1
    start_portfolio_i = start_portfolio + timedelta(days=1)

    portfolio_value = [0] * (time_window_days_minus1)
    portfolio_value.append(initial_value)
    portfolio_return_index = time_window_days_minus1

    while start_portfolio_i <= end_date:

        ret = returns.loc[start_portfolio_i]
        ret = ret.loc[i]

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

    returns['portfolio_value '+i] = portfolio_value


"""
#Descriptive Statistics of all the logarithmic returns of Cryptocurrencies
ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median','0.1', '0.25', '0.75', '0.9', 'volatility', 'dr']
table = [0]*(len(ds))
table = pd.DataFrame([table],columns=ds)
table['crypto'] = 'initialize'
table=table.set_index(table['crypto'])

print(table)

for i in cryptos:

    ret = df.loc[start_date:end_date, i]
    print(ret)

    start_portfolio_i = start_date
    portfolio_return_index = time_window_days_minus1

    while start_portfolio_i <= end_date:

        x = ret.loc[start_portfolio_i]

        if x == 0 or math.isnan(x):

            start_portfolio_i = start_portfolio_i + timedelta(days=1)

        else:
            start_analysis = start_portfolio_i + timedelta(days=1)
            break

    ret = returns.loc[start_analysis:end_date, i]
    ret = ret*100

    x = st.describe(ret)
    print(x)

    desc = list()
    z = [0,1,2,3,4,5]

    for j in z:

        if j ==1:
            value = x[j]
            minimum = value[0]
            desc.append(minimum)
            maximum = value[1]
            desc.append(maximum)
        else:
            value = x[j]
            desc.append(value)

    median = statistics.median(ret)
    desc.append(median)
    print(median)
    print(desc)

    q = [0.1, 0.25, 0.75, 0.9]
    quantiles = ret.quantile(q)

    for k in q:
        h = quantiles[k]
        desc.append(h)
    #shapiro = st.shapiro(ret)
    #print (shapiro)
    #sh = shapiro[1]
    #print (sh)
    #desc.append(sh)

    ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median','0.1', '0.25', '0.75', '0.9']#'Shapiro-Wilk test p-value'

    f= pd.DataFrame([desc], columns=ds)
    f['crypto'] = i
    f=f.set_index(f['crypto'])
    print(f)

    table = [table, f]
    table = pd.concat(table)#ignore_index=False

"""
"""
#Calculate Portfolio Statistics with Prices
ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median', 'Sharpe Ratio','VaR', 'Expected Shortfall','volatility']
table = [0]*(len(ds))
table = pd.DataFrame([table],columns=ds)
table['crypto'] = 'initialize'
table=table.set_index(table['crypto'])

for i in cryptos:

    ret = df.loc[start_portfolio:end_date, i]
    print(ret)

    start_portfolio_i = start_portfolio #+ timedelta(days=1)
    portfolio_return_index = time_window_days_minus1

    while start_portfolio_i <= end_date:

        x = ret.loc[start_portfolio_i]

        if x == 0 or math.isnan(x):

            start_portfolio_i = start_portfolio_i + timedelta(days=1)

        else:
            start_analysis = start_portfolio_i + timedelta(days=1)
            break

    ret = returns.loc[start_analysis:end_date, i]

    hist = ret.to_numpy()
    print(hist)
    print(ret)

    x = st.describe(ret)
    print(x)

    desc = list()
    z = [0,1,2,3,4,5]

    for j in z:

        if j ==1:
            value = x[j]
            minimum = value[0]
            desc.append(minimum)
            maximum = value[1]
            desc.append(maximum)
        else:
            value = x[j]
            desc.append(value)

    median = statistics.median(ret)
    desc.append(median)
    print(median)
    print(desc)

    sigma_h = np.sqrt(x[3])
    mu_h = x[2]
    sharpe = (mu_h-rf)/sigma_h
    desc.append(sharpe)

    alpha = 0.01 #95% confidence level
    VaR_n = norm.ppf(1 - alpha) * sigma_h - (mu_h-rf)
    desc.append(VaR_n)

    CVaR_n = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sigma_h - (mu_h-rf)
    desc.append(CVaR_n)

    desc.append(sigma_h)

    #shapiro = st.shapiro(ret)
    #print (shapiro)
    #sh = shapiro[1]
    #print (sh)
    #desc.append(sh)

    ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median', 'Sharpe Ratio', 'VaR', 'Expected Shortfall','volatility']#'Shapiro-Wilk test p-value'

    f= pd.DataFrame([desc], columns=ds)
    f['crypto'] = i
    f=f.set_index(f['crypto'])
    print(f)

    table = [table, f]
    table = pd.concat(table)#ignore_index=False

    print(table)


"""


#Statistics of Portfolios
#cc = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/PR_norm.xlsx')
#cc = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/CC_norm.xlsx')
#cc = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/CCRET.xlsx')
cc = pd.read_excel(r'/Users/gmmtakane/Desktop/Thesis/Port_month_ret.xlsx')
cc = cc.set_index(pd.to_datetime(cc['date']), drop=True)
del cc['date']
lst = list(cc.columns)

ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median', 'Sharpe Ratio','VaR', 'Expected Shortfall','volatility','0.1', '0.05','0.25', '0.75', '0.9','dr']
table = [0]*(len(ds))
table = pd.DataFrame([table],columns=ds)
table['crypto'] = 'initialize'
table=table.set_index(table['crypto'])

for i in lst:

    ret = cc.loc[start_portfolio:end_date, i]
    print(ret)

    start_portfolio_i = start_portfolio + timedelta(days=1)
    portfolio_return_index = time_window_days_minus1

    while start_portfolio_i <= end_date:

        x = ret.loc[start_portfolio_i]

        if x == 0 or math.isnan(x):

            start_portfolio_i = start_portfolio_i + timedelta(days=1)

        else:
            start_analysis = start_portfolio_i
            break


    #start_p = start_analysis + timedelta(days=1)



    ret = cc.loc[start_analysis:end_date, i]
    print(ret)

    hist = ret.to_numpy()
    print(hist)
    print(ret)

    x = st.describe(ret)
    print(x)

    desc = list()
    z = [0,1,2,3,4,5]

    for j in z:

        if j ==1:
            value = x[j]
            minimum = value[0]
            desc.append(minimum)
            maximum = value[1]
            desc.append(maximum)
        else:
            value = x[j]
            desc.append(value)

    median = statistics.median(ret)
    desc.append(median)
    print(median)
    print(desc)

    sigma_h = np.sqrt(x[3])
    mu_h = x[2]
    sharpe = (mu_h-rf)/sigma_h
    sharpe = sharpe * np.sqrt(365)
    desc.append(sharpe)

    alpha = 0.01 #95% confidence level
    VaR_n = -norm.ppf(1 - alpha) * sigma_h - (mu_h-rf)
    desc.append(VaR_n)

    CVaR_n = -alpha ** -1 * norm.pdf(norm.ppf(alpha)) * sigma_h - (mu_h-rf)
    desc.append(CVaR_n)

    desc.append(sigma_h)

    q=[0.1, 0.05, 0.25, 0.75, 0.9]
    quantile = ret.quantile(q)
    print(quantile)

    for k in q:

        h=quantile[k]
        desc.append(h)


    daily_r = math.exp(x[2]*x[0]/100)
    desc.append(daily_r)

    #shapiro = st.shapiro(ret)
    #print (shapiro)
    #sh = shapiro[1]
    #print (sh)
   # desc.append(sh)

    ds = ['N','Minimum','Maximum','Mean','Variance','Skewness','Kurtosis','Median', 'Sharpe Ratio', 'VaR', 'Expected Shortfall','volatility','0.1', '0.05','0.25', '0.75', '0.9','dr']#'Shapiro-Wilk test p-value'

    f= pd.DataFrame([desc], columns=ds)
    f['portfolio'] = i
    f=f.set_index(f['portfolio'])
    print(f)

    table = [table, f]
    table = pd.concat(table)#ignore_index=False

    print(table)

#Save the dataframes to Excel file
table.to_excel(r'/Users/gmmtakane/Desktop/Thesis/OPM' + '.xlsx')
#returns.to_excel(r'/Users/gmmtakane/Desktop/Thesis/ret performance2' + '.xlsx')
