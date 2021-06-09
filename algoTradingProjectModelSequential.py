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
ml['returns_2021'] = df[df['year'] == 2021].groupby('Name').apply(calculate_returns)
#print(ml)
print(ml.columns)

ml['target'] = ml['returns_2021'] >= 0.08



print('Target Counts')
print(ml['target'].value_counts())
ml = ml.dropna()


# add some features to model
cols_1 = ['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
            '2019_5', '2019_6', '2019_7', '2019_8', '2019_9', '2020_1', '2020_10',
            '2020_11', '2020_12', '2020_2', '2020_3',  '2020_5', '2020_6',
             '2020_8', '2020_9','returns_2019']
ml['returns_2019'] = df[df['year'] == 2019].groupby('Name').apply(calculate_returns)
ml['returns_2020'] = df[df['year'] == 2020].groupby('Name').apply(calculate_returns)

ax=ml[ml['target']==1].plot(x='returns_2019',y='returns_2020',kind = 'scatter',color='green')
ax=ml[ml['target'] == 0].plot(x='returns_2019',y='returns_2020',kind = 'scatter',color='red',ax=ax)
plt.show()

#searching for other features for model
#if i want the max of each share in 2019
year_2019_months=['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
                  '2019_5', '2019_6', '2019_7', '2019_8', '2019_9']
year_2020_months=['2020_1', '2020_10', '2020_11', '2020_12', '2020_2', '2020_3', '2020_4',
                  '2020_5', '2020_6', '2020_7', '2020_8', '2020_9']

ml['max_months_2019']=ml.loc[:,year_2019_months].apply(lambda x: x.max(),axis=1)#add this to ml
ml['min_months_2019']=ml.loc[:,year_2019_months].apply(lambda x: x.min(),axis=1)#add this to ml
ml['mean_months_2019']=ml.loc[:,year_2019_months].apply(lambda x: x.mean(),axis=1)#add this to ml
ml['std_months_2019']=ml.loc[:,year_2019_months].apply(lambda x: x.std(),axis=1)#add this to ml


ml['max_months_2020']=ml.loc[:,year_2020_months].apply(lambda x: x.max(),axis=1)#add this to ml
ml['min_months_2020']=ml.loc[:,year_2020_months].apply(lambda x: x.min(),axis=1)#add this to ml
ml['mean_months_2020']=ml.loc[:,year_2020_months].apply(lambda x: x.mean(),axis=1)#add this to ml
ml['std_months_2020']=ml.loc[:,year_2020_months].apply(lambda x: x.std(),axis=1)#add this to ml

ml['vol_at_2019_1'] = df[df['year_month'] =='2019_1'].groupby('Name')['volume'].mean()
ml['vol_at_2019_12'] = df[df['year_month'] =='2019_12'].groupby('Name')['volume'].mean()
ml['vol_at_2020_1'] = df[df['year_month'] =='2020_1'].groupby('Name')['volume'].mean()
ml['vol_at_2020_12'] = df[df['year_month'] =='2020_12'].groupby('Name')['volume'].mean()


ml['norm_vol_at_2019_12'] = ml['vol_at_2019_12'] / ml['vol_at_2019_1']
ml['norm_vol_at_2020_1'] = ml['vol_at_2020_1'] / ml['vol_at_2019_1']
ml['norm_vol_at_2020_12'] = ml['vol_at_2020_12'] / ml['vol_at_2019_1']

cols_2 = ['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
        '2019_5', '2019_6', '2019_7', '2019_8', '2019_9', '2020_1', '2020_10',
        '2020_11', '2020_12', '2020_2', '2020_3',  '2020_5', '2020_6',
        '2020_8', '2020_9','returns_2019','max_months_2019','min_months_2019','mean_months_2019',
          'std_months_2019']

cols_3 = ['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
          '2019_5', '2019_6', '2019_7', '2019_8', '2019_9', '2020_1', '2020_10',
          '2020_11', '2020_12', '2020_2', '2020_3',  '2020_5', '2020_6',
          '2020_8', '2020_9','returns_2019','max_months_2020','min_months_2020','mean_months_2020',
          'std_months_2020']

cols_4 = ['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3', '2019_4',
          '2019_5', '2019_6', '2019_7',  '2019_9', '2020_1', '2020_10',
          '2020_11', '2020_12', '2020_2', '2020_3',  '2020_5', '2020_6',
          '2020_8','min_months_2019','mean_months_2019',
          'max_months_2020','min_months_2020','mean_months_2020',
          'std_months_2020']

cols_5 = ['2019_1', '2019_10', '2019_11', '2019_12', '2019_2', '2019_3',
           '2019_6', '2019_7',  '2019_9', '2020_1', '2020_10',
           '2020_12',   '2020_5', '2020_6',
          '2020_8','min_months_2019','mean_months_2019',
          'max_months_2020','min_months_2020',
          'std_months_2020']


cols_6 =['2020_1', '2020_12','2019_5','2019_3' , 'returns_2019','2020_7', '2019_5', '2020_4',
         'returns_2020',  'max_months_2020','2019_10', '2020_5', '2019_3', '2019_11',
         'mean_months_2020',  'min_months_2019','norm_vol_at_2019_12',
         'norm_vol_at_2020_1','norm_vol_at_2020_12']


#model
clf = RandomForestClassifier(random_state = 4)
ml = ml.sample(frac=1,random_state=41)
X = ml[cols_6]
print(X.columns)

y = ml['target']
print(y)

#split data to train and test
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

#train the model
clf.fit(X_train,y_train)

#predict on X_test
y_pred = clf.predict(X_test)

#calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix')
print(cm)


all_etfs=X_test.index
print(all_etfs)



print(100*ml.loc[all_etfs]['returns_2021'].mean())
print(100*ml.loc[X_test[y_pred].index]['returns_2021'].mean())
print(100*ml.loc[X_test[~y_pred].index]['returns_2021'].mean())

print(clf.feature_importances_)
print(X_test.columns)
ft = pd.DataFrame(zip(X_test.columns,clf.feature_importances_))

ft.sort_values(1,ascending=False).plot(x=0,y=1,kind='bar')
print(ft.sort_values(1,ascending=False).tail(10)[0])# they are the worst features that be deleted and chek again
plt.show()

#addimg the best ft_values 2020_12,2019_9  to the columns





