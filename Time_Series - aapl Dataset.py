# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:32:00 2024

@author: Priyanka
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df= pd.read_csv("C:/Data Set/aapl.csv")
df
df.head(2)

type(df.Date[0])

df= pd.read_csv("C:/Data Set/aapl.csv",parse_dates=["Date"])
df.head(2)

type(df.Date[0])

df= pd.read_csv("C:/Data Set/aapl.csv",parse_dates=["Date"],index_col='Date')
df.index
df['2017-01']
df['2017-01-04']
df['2017-01'].Close.mean()
df['2017-06'].head()
df['2017-06'].Close.mean()

df['2017-01-08':'2017-01-03']


df['Close'].resample('M').mean().head()

df['2016-07']

%matplotlib.inline
df["Close"].plot()

df['Close'].resample('Q').mean().plot()
df['Close'].resample('Q').mean().plot(kind='bar')
