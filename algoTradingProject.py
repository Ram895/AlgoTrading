import numpy as np
import pandas as pd
import requests
import math


stocks=pd.read_csv('D:/Networks/security/data_sience/algo_trading/data_seience_endProjectWork_algoTrading/eft5y.csv')

#change the date type
stocks['dt'] = pd.to_datetime(stocks['date'])
#change name column to Name
stocks.columns=['date', 'open', 'low', 'close', 'volume', 'Name', 'dt']
print(stocks.head())
df=stocks.copy()

df['days'] = (df['dt'] - df['dt'].min()).dt.days.astype(int)

TOTAL_TRAIN_TIME = 50
TOTAL_TEST_TIME = 100
NUMBER_OF_STOCKS = 10


def calculate_returns(x):
    return 100*(x['close'].iloc[-1] - x['open'].iloc[0])/x['open'].iloc[0]

# What was the mean return of all the ETF's 133  in the test dataset
# What was the mean return of all the top 10 ETFs in the test dataset

TOTAL_TRAIN_TIME = 50
TOTAL_TEST_TIME = 100
NUMBER_OF_STOCKS = 10
def calculate_returns(x):
    return 100*(x['close'].iloc[-1] - x['open'].iloc[0])/x['open'].iloc[0]


delta = 0
start_train_day = delta
end_train_day = start_train_day + TOTAL_TRAIN_TIME
start_test_day = end_train_day + 1
end_test_day = start_test_day + TOTAL_TEST_TIME
train = df[(df['days'] >= start_train_day) & (df['days'] < end_train_day)]
test = df[(df['days'] >= start_test_day) & (df['days'] < end_test_day)]
good_stock = train.groupby('Name').apply(calculate_returns).sort_values(ascending=False).head(NUMBER_OF_STOCKS).index




print('\n')
print(good_stock)
print('\n')
print('The returns of all ETF"s is : ',test.groupby('Name').apply(calculate_returns).mean())
print('The returns of the 10 good ETF"s is : ',test[test['Name'].isin(good_stock)].groupby('Name').apply(calculate_returns).mean())
