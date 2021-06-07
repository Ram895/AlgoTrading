import numpy as np
import pandas as pd
import requests
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



stocks=pd.read_csv('D:/Networks/security/data_sience/algo_trading/data_seience_endProjectWork_algoTrading/eft5y.csv')
print(stocks.head())

#change the date type
stocks['dt'] = pd.to_datetime(stocks['date'])
#change name column to Name
stocks.columns=['date', 'open', 'low', 'high','close', 'volume', 'Name', 'dt']
print(stocks.head())
df=stocks.copy()

df['year'] = df['dt'].dt.year
df['month'] = df['dt'].dt.month
df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype('str')
df = df.sort_values('dt')


def calculate_returns(df):
    return (df['close'].iloc[-1] - df['open'].iloc[0] ) / df['open'].iloc[0]

ml = pd.DataFrame()
ml['returns_2019'] = df[df['year'] == 2019].groupby('Name').apply(calculate_returns)
ml['returns_2020'] = df[df['year'] == 2020].groupby('Name').apply(calculate_returns)
ml['returns_2021'] = df[df['year'] == 2021].groupby('Name').apply(calculate_returns)

a = df[df['year'].isin([2019,2020])].groupby(['Name','year_month']).apply(calculate_returns)
ml=a.reset_index().pivot_table(index='Name',columns='year_month',values=0)
print(ml)

ml['returns_2021'] = df[df['year'] == 2021].groupby('Name').apply(calculate_returns)

ml['target'] = ml['returns_2021'] >= 0
print(ml)


print('Target Counts')
print(ml['target'].value_counts())
ml = ml.dropna()


# add some features to model
cols = ['returns_2020','returns_2019','2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
            '2019_5', '2019_6', '2019_7', '2019_8', '2019_9', '2020_1', '2020_10',
            '2020_11', '2020_12', '2020_2', '2020_3', '2020_4', '2020_5', '2020_6',
            '2020_7', '2020_8', '2020_9']

ax=ml[ml['target']==1].plot(x='returns_2019',y='returns_2020',kind = 'scatter',color='green')
ax=ml[ml['target'] == 0].plot(x='returns_2019',y='returns_2020',kind = 'scatter',color='red',ax=ax)
plt.show()




#model
clf = RandomForestClassifier()
ml = ml.sample(frac=1)
X = ml[cols]
y = ml['target']

#split data to train and test
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3)

#train the model
clf.fit(X_train,y_train)

#predict on X_test
y_pred = clf.predict(X_test)

#calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix')
print(cm)
print('You have overfitting check again')

all_etfs=X_test.index
print(len(all_etfs))



print(100*ml.loc[all_etfs]['returns_2021'].mean())
print(100*ml.loc[X_test[y_pred].index]['returns_2021'].mean())
print(100*ml.loc[X_test[~y_pred].index]['returns_2021'].mean())


print(ml.loc[X_test[y_pred].index, ])










