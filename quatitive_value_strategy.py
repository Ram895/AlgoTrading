# https://w1615205118-nvx971826.slack.com/archives/D01R407DLKT/p1623074127001500   מצגת של אמיר קטורזה
import numpy as np
import pandas as pd
import requests
#import xlswriter
from scipy import stats
import math

#importing our list of stocks & API Token
IEX_CLOUD_API_TOKEN = 'Tpk_059b97af715d417d9f49f50b51b1c448'

stocks=pd.read_csv('D:/Networks/security/data_sience/algo_trading/sp_500_stocks.csv')
stocks=stocks.dropna()
stocks=stocks[['Symbol','Name']]
print(stocks)

#Making Our First API Call
symbol='MMM'
api_url=f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data=requests.get(api_url)
print(data)
data=data.json()
print(data)

#Parsing Our API Call
price = data['latestPrice']
pe_ratio = data['peRatio']
print(price)
print(pe_ratio)

#Executing A Batch API Call & Building Our DataFrame
def chunks(lst,n):#generator function
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst),n):
        yield lst[i:i+n]

symbol_groups =list(chunks(stocks['Symbol'],100))# generator
#print(symbol_groups)
symbol_strings=[]
for i in range(0,len(symbol_groups)):
    symbol_strings.append(",".join(symbol_groups[i]))

print(symbol_strings)
my_columns = ['Symbol','Price','Price to Earnings Ratio','Number of Shares to Buy']

final_dataframe = pd.DataFrame(columns=my_columns)
for symbol_string in symbol_strings:
    batch_api_call_url = api_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    # print(data)
    i = 0
    for symbol in symbol_string.split(','):
        final_dataframe = final_dataframe.append(
            pd.Series(
                [
                    symbol,
                    data[symbol]['quote']['latestPrice'],
                    data[symbol]['quote']['peRatio'],
                    'N/A'
                ],
                index=my_columns),
            ignore_index=True
        )
print(final_dataframe)

#Removing Glamour Stocks-(the opposite of a "value stock")

final_dataframe.sort_values('Price to Earnings Ratio', inplace=True )
final_dataframe = final_dataframe[final_dataframe['Price to Earnings Ratio']>0]
final_dataframe = final_dataframe[:50].reset_index()
final_dataframe.drop('index' ,axis=1, inplace=True)
print(final_dataframe)
#del df['column_name']

#Calculating the Number of Shares to Buy
def protfolio_input():
    global protfolio_size,val
    protfolio_size=input('Enter the value of your porfolio:')
    try:
        val = float(protfolio_size)
        print(val)
    except ValueError:
        print('that"s not a number! \nPlease try again:')
        protfolio_size=input('Enter the value of your porfolio:')
        val=float(protfolio_size)

protfolio_input()

posizion_size=float(val/len(final_dataframe.index))
print(posizion_size)
for row in final_dataframe.index:
    #print(row)
    final_dataframe.loc[row,'Number of Shares to Buy']=math.floor(posizion_size/final_dataframe.loc[row,'Price'])

print(final_dataframe)

print('posizion_size is: ',posizion_size)
print(f'Protfolio has {len(final_dataframe.index)} Shares')

#print(math.floor(final_dataframe['Number of Shares to Buy'].values))

#Building a Better(and More Realistic) Momentum Strategy
symbol='AAPL'
batch_api_call_url = api_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
data = requests.get(batch_api_call_url).json()
print(data)
#Price-to-earning ratio
pe_ratio = data[symbol]['quote']['peRatio']

#Price-to-book ratio
pb_ratio = data[symbol]['advanced-stats']['priceToBook']

#Price-to-sales ratio
ps_ratio = data[symbol]['advanced-stats']['priceToSales']

#Enterprice Value divided by Earning Before Interest, Taxes, Depreciation, and Amortization
enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
ebitda = data[symbol]['advanced-stats']['EBITDA']

ev_to_ebitda = enterprise_value/ebitda

#Enterprice Value divided by Gross Profit (EV/GP)
gross_profit = data[symbol]['advanced-stats']['grossProfit']
ev_to_gross_profit = enterprise_value/gross_profit

rv_columns=[
    'Ticker',
    'Price',
    'Number of shares to Buy',
    'Price-to-earning ratio',
    'PE Percentile',
    'Price-to-book ratio',
    'PB Percentile',
    'Price-to-sales ratio',
    'PS Percentile',
    'EV/EBITDA',
    'EV/EBITDA Percentile',
    'EV/GP',
    'EV/GP Percentile',
    'RV Score'
]

rv_dataframe = pd.DataFrame(columns=rv_columns)
print(rv_dataframe)

def chunks(lst,n):#generator function
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst),n):
        yield lst[i:i+n]


symbol_groups = list(chunks(stocks['Symbol'], 100))  # generator
# print(symbol_groups)
symbol_strings = []
for i in range(0, len(symbol_groups)):
    try:
        # print(symbol_group)
        symbol_strings.append(",".join(symbol_groups[i]))
    except:
        print(symbol_groups[i])



for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    #print(data)
    
    for symbol in symbol_string.split(','):
        # print(symbol)
        final_dataframe = final_dataframe.append(
            pd.Series(
                [
                    symbol,
                    data[symbol]['quote']['latestPrice'],
                    data[symbol]['quote']['marketCap'],
                    'N/A'
                ],
                index=my_columns),
            ignore_index=True
        )

final_dataframe


