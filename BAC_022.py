#!/usr/bin/env python
# coding: utf-8

# In[71]:


#install libraries and dependencies
import numpy as np
import pandas as pd


# In[72]:


#Load datasets
df = pd.read_csv('/Users/Celia/desktop/Current course/data enginering & mining/datasets/BAC.csv')
df.head(10)


# In[73]:


#install libraries and dependencies
from matplotlib import pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[74]:


#data cleaning, check null or not
df.isnull().sum()


# In[75]:


#Count data shape
df.shape


# In[76]:


from pandas.plotting import lag_plot
from pandas import datetime


# In[9]:


#plot the close price evolution over time in terms of month
df1=df.groupby(['Date'])['Close'].mean()
df1.plot(figsize=(12,8), title= 'Closing Prices(in Month)', fontsize=14);


# In[77]:


#Analyse the autocorrelation plot of the “Close” feature with respect to a fixed lag of 5.
plt.figure(figsize=(5,5))
lag_plot(df['Close'], lag=5)
plt.title('BAC Autocorrelation plot')
plt.show()

#The results shown in the below figure confirmed the ARIMA would have been a good model to be applied to this type of data.


# In[78]:


#Only one column neede to train and test dataset
filterdf = df['Close']

#Training the dataset with more no of records
X_train = filterdf[:2400]

#Use some of the records to test
X_test = filterdf[2401:]


# In[88]:


#define function for ADF test to test stationary
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
def adf_test_stationary(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='orange', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller Test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#apply adf test on the series
#P value > .95 and increasing mean & SD inferring the series is non-stationary

adf_test_stationary(X_train)


# In[79]:


#separate Trend and Seasonality from the time series
result = seasonal_decompose(X_train, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


# In[96]:


#taking log of the series,find the rolling average(past 12 months)
plt.rcParams['figure.figsize'] = (10, 6)
df_log = np.log(X_train)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average')
plt.plot(std_dev, color ="black", label = "Standard Deviation")
plt.plot(moving_avg, color="red", label = "Mean")
plt.legend()
plt.show()


# In[90]:


#Use error function--Sysmmetric Mean Absolute Percentage Error(SMAPE)
def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) +       np.abs(y_true))))


# In[91]:


#Use error function--Mean Squared Error (MSE)
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[ ]:


#Created ARIMA model to be used for implementation. Set p=5, d=1 and q=0 as parameters
train_ar = train_data['Close'].values
test_ar = test_data['Close'].values
history = [x for x in train_ar]
print(type(history))
predictions = list()
for t in range(len(test_ar)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test_ar[t]
    history.append(obs)
error = mean_squared_error(test_ar, predictions)
print('Testing Mean Squared Error: %.3f' % error)
error2 = smape_kun(test_ar, predictions)
print('Symmetric mean absolute percentage error: %.3f' % error2)


# In[66]:


#divide the data into a training (80%) and test (20%) set
train_data, test_data = df[0:int(len(df)*0.8)], df[int(len(df)*0.8):]
plt.figure(figsize=(12,7))
plt.title('BAC Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(df['Close'], 'blue', label='Training Data')
plt.plot(test_data['Close'], 'green', label='Testing Data')
plt.xticks(np.arange(0, 2589, 500), df['Date'][0:2589:500])
plt.legend()


# In[ ]:


#plot the training,test and predicted prices against time
plt.figure(figsize=(12,7))
plt.plot(df['Close'], 'green', color='blue', label='Training Data')
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', 
         label='Predicted Price')
plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
plt.title('BAC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(0, 2589, 500), df['Date'][0:2589:500])
plt.legend()


# In[69]:


#Zoomed in version of Prediction vs Actual Price Comparison
plt.figure(figsize=(12,7))
plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_data.index, test_data['Close'], color='red', label='Actual Price')
plt.legend()
plt.title('BAC Prices Prediction')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.xticks(np.arange(2000, 2589, 100), df['Date'][2000:2589:100])
plt.legend()


# In[ ]:




