# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:23:51 2024

@author: Priyanka
"""

import pandas as pd
import numpy as np
walmart=pd.read_csv("C:\Data Set\Walmart_Footfalls_Raw.csv")


#----------------------------------------- Pre Processing --------------------------------

walmart["t"] = np.arange(1,160)

walmart["t_square"]= walmart["t"] * walmart["t"]
walmart["log_footfalls"] = np.log(walmart["Footfalls"])
walmart.columns
#month=['Jan,'Feb','Apr,'May','Jun','July','Aug','Sep','Oct','Nov','Dec']
#in walmart data we have jan-1991 in 0th column we needd only 
#example jan from each cell

p=walmart["Month"][0]
#before we will extract let us create new column called month
p[0:3]
#'Jan'
walmart['months']=0

for i in range(159):
    p=walmart["Month"][i]
    walmart['months'][i]=p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(walmart['months']))

walmart1= pd.concat([walmart,month_dummies],axis=1)

#------------------------------Visualization - time plot----------------------------

walmart1.Footfalls.plot()

#-----------------------------------Data Partition-----------------------------------

Train = walmart1.head(147)
Test=walmart1.tail(12)
#to change the index values in pandas data frame
#Test.set_index(np.arange(1,13))


#---------------------------------------------Linear-------------------------------------

import statsmodels.formula.api as smf
linear_model = smf.ols('Footfalls ~ t',data = Train).fit()
pred_linear= pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear
#209.92559265462594


#-----------------------------------Exponential Model----------------------------------

Exp=smf.ols('log_footfalls ~ t',data=Train).fit()
pred_Exp=pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp=np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp
#217.05263566813454


#---------------------------------------Quadratic model--------------------------------

Quad= smf.ols('Footfalls ~ t + t_square',data= Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad
#137.1546274135642



#--------------------------------Additive Seasonality Quadratic Trend----------------------

add_sea=smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()

pred_add_sea=pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))

rmse_add_sea= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea
#264.6643900568772


#---------------------------------Multiplicative Seasonality-----------------------------

Mul_sea=smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()

pred_Mul_sea=pd.Series(Mul_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))

rmse_Mul_sea= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.exp(pred_Mul_sea))**2))
rmse_Mul_sea
#268.1970325309192


#------------------------------Additive Seasonality Quadratic Trend--------------------------
add_sea_Quad=smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()

pred_add_sea_quad=pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))

rmse_add_sea_quad= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
#50.60724584169114


#-----------------------Multiplicative Seassonality Linear trand----------------------

Mul_Add_sea=smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()

pred_Mul_add_sea=pd.Series(Mul_Add_sea.predict(Test))

rmse_Mul_add_sea= np.sqrt(np.mean((np.array(Test['Footfalls']) - np.exp(pred_Mul_add_sea))**2))
rmse_Mul_add_sea
#

#----------------------------------Testing--------------------------------------

data ={"MODEL": pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_Mul_sea','rmse_add_sea_quad','rmse_Mul_add_sea']),
       "RMSE_Values": pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_Mul_sea,rmse_add_sea_quad,rmse_Mul_add_sea])}
table_rmse= pd.DataFrame(data)
table_rmse
'''
MODEL  RMSE_Values
0        rmse_linear   209.925593
1           rmse_Exp   217.052636
2          rmse_Quad   137.154627
3       rmse_add_sea   264.664390
4       rmse_Mul_sea   268.197033
5  rmse_add_sea_quad    50.607246
6   rmse_Mul_add_sea   172.767268

'''

#---------------------------Testing----------------------------------
#rmse_add_sea has the least values among the models prepared so far
predict_data = pd.read_excel("C:\Data Set\Predict_new.xlsx")

model_full = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=walmart1).fit()

pred_new =pd.Series(model_full.predict(predict_data))
pred_new
'''
0     2193.807626
1     2229.969736
2     2200.670308
3     2311.293957
4     2356.071452
5     2036.848947
6     2187.241826
7     2181.480859
8     2234.104508
9     1999.997498
10    1972.995363
11    2280.493228
'''


predict_data["forecasted_Footfalls"] = pd.Series(pred_new)


#---------------------Autoregression Model (AR)---------------------
#Calculating Residuals from best model applied on full data
#AV- FV

full_res= walmart1.Footfalls - model_full.predict(walmart1)

#ACF Plot on Residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res,lags=12)

#ACF is an (complete) auto correlation function gives values
#of auto correlation of any time series with its lagged values

#PACF is a partial auto correlation function
#it finds correlation of present with lags of the residuals of the

tsa_plots.plot_pacf(full_res,lags=12)

#Alternative approach for ACF plot
#from panads.plotting import autocorrelation_plot
#autocorrection_pyplot.show()


#---------------------AR Model-----------------------------
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags=[1])
#model_ar = AutoReg(Train_res, lags=[12])
model_fit = model_ar.fit()

print('Coeficients: %s' % model_fit.params)
'''
Coeficients: const   -0.543345
y.L1     0.638663
'''

pred_res=model_fit.predict(start=len(full_res),end=len(full_res)+len(predict_data))

pred_res.reset_index(drop=True,inplace=True)

#The final Predictions using ASQT and AR(1) model

final_pred = pred_new + pred_res
final_pred

'''
0     2164.917162
1     2210.975131
2     2187.995818
3     2302.655889
4     2350.011295
5     2032.435206
6     2183.879589
7     2178.790179
8     2231.842726
9     1998.009637
10    1971.182445
11    2278.792040
12            NaN
'''
