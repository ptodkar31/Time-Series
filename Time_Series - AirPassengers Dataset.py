# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:29:10 2024

@author: Priyanka
"""

'''
International Airline Passengers prediction
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('dark_background')
#load the data set
df=pd.read_csv("C:\Data Set\AirPassengers.csv")
df.columns
df.head
df= df.rename({'#Passengers':'Passengers'},axis=1)
print(df.dtypes)

#Months is test and passangers is int
#Now let us convert into date and time
df['Month']= pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month',inplace=True)

plt.plot(df.Passengers)
#There is incresing trennd and it has got seasonality

#Is the data Sationary?
#Dickey-Fuller test 
from statsmodels.tsa.stattools import adfuller
adf , pvalue, usedlag_, nobs_, critical_value, icbest_ = adfuller(df)

print("pvalue = ", pvalue," if above 0.05, data is not sessonality")
#since data is not stationary we may need SARIMA and not just ARIMA
#now let us extract the year and month from the data and time column

df['year']=[d.year for d in df.index]
df['month']=[d.strftime('%b') for d in df.index]
years = df['year'].unique()

#plot yearly and monthly values as boxplot
sns.boxplot(x='year',y='Passengers',data=df)
#No. of passangers are going up year by year
sns.boxplot(x='month',y='Passengers',data=df)
#Over all there is higher trend in July and August Compared to rest of the

#Extract and plot trend seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose

decompose = seasonal_decompose(df['Passengers'],model='additive')
#Additive time series
#values = Base Lever +Trend 

trend=decompose.trend
seasonal=decompose.seasonal
residual=decompose.residual

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passengers'],label='Original',color='Yellow')
plt.legend(loc="upper left")
plt.subplot(412)
plt.plot(trend,label='Trend',color='Yellow')
plt.legend(loc="upper left")
plt.subplot(413)
plt.plot(seasonal,label='Seasonal',color='Yellow')
plt.legend(loc="upper left")
plt.subplot(414)
plt.plot(residual,label='Residual',color='Yellow')
plt.legend(loc="upper left")
plt.show()

'''Trend is going up from 950s to 60s
It is highly seasional showing peaks at particular interval
This helps to select specific prediction model
'''


#Autocorrelation
#values are not correlation with x_aixs but with its lags
#meaning yesterdays values is depend on day before yesterday so on  so
#Autocorrelation is simply the correlation of a series with its own lags
#plot lag on x axis and correlation on y axis
#any correlation above confidence lnes are statistically significant

from statsmodels.tsa.stattools import acf

acf_144=acf(df.Passengers, nlags=144)
plt.plot(acf_144)

#Auto corrlation above zero means postitive correation and below as negative
#Obtain the same but with single line and more info

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)

#Any lag before 40 has postive correlation
#Horizonatl bands indicates 95% and 99% (dashed ) confidence band
