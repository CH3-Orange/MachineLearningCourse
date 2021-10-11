# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

test=pd.read_csv("test.csv")
# test.head()
train =pd.read_csv("train.csv")
# train.head()
#describe函数可以统计数据信息，不加上include=all的话默认统计数值型数据
# train.describe(include="all")
# test.describe(include="all")
# print("train: \n",train.describe(include="all"))
train.info()
train.drop(['Id'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)
train.info()
plt.figure()
plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

train[train["GrLivArea"]>4000 and train["SalePrice"]<300000]

#train.drop(train[train["GrLivArea"]])