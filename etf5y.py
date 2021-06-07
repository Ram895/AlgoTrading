from datetime import datetime
import backtrader as bt
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import secrets

# %matplotlib inline
stocks = pd.read_csv('D:/Networks/security/data_sience/algo_trading/spdr-product-data-us-en.csv')
stocks = stocks.dropna()
# print(stocks)
tickers = stocks['Ticker']

tickers = tickers.to_numpy()
# print(tickers)


range='5y'
symbols,closes,lows,highs,opens,volumes,prices,dates=[],[],[],[],[],[],[],[]
for symbol in tickers:
    print('symbol is : ',symbol)
    api_url = f'https://sandbox.iexapis.com/stable/stock/twtr/chart/dynamic?symbols={symbol}&types=price,quote&range=5y&token={secrets.IEX_CLOUD_API_TOKEN}'
    data = requests.get(api_url).json()
    print(data)

    for i,etf in enumerate(data):
        close = data[i]['close']
        low = data[i]['low']
        high = data[i]['high']
        open = data[i]['open']
        date=data[i]['date']
        volume=data[i]['volume']
        volumes.append(volume)
        dates.append(date)
        symbols.append(symbol)
        closes.append(close)
        highs.append(high)
        opens.append(open)
        lows.append(low)
    print(len(highs))
    print(len(lows))
    print(len(symbols))
    print(len(opens))
    print(len(closes))
    print(len(dates))
    print(len(volumes))




d={'date':dates,'open':opens,'low':lows,'high':highs,'close':closes,'volume':volumes,'name':symbols}
df=pd.DataFrame(d)
df.to_csv('D:/Networks/security/data_sience/algo_trading/data_seience_endProjectWork_algoTrading/eft5y.csv',index=False)
print(df)
