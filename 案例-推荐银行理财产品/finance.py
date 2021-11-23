# -*- coding: utf-8 -*-
#finance.py
##推荐银行理财产品
"""
Created on Sat Nov 20 09:19:59 2021

@author: Orange
"""
# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#导入数据
data= pd.read_csv("./credit_card.csv")

#统计缺失数据
def calcNull(data):
    nullSum=data.isnull().sum()
    nullSum=nullSum.drop(nullSum[nullSum.values==0].index)
    return nullSum
missing_data=calcNull(data)

#删除缺失数据行
data.drop(data[ np.isnan(data['CREDIT_LIMIT']) ].index,axis=0,inplace=True)
data.drop(data[ np.isnan(data['MINIMUM_PAYMENTS']) ].index,axis=0,inplace=True)

#删除CUST_ID列
data.drop(["CUST_ID"],axis=1,inplace=True)

#绘制热力图
import seaborn as sns
corr = data.corr()
plt.figure(figsize=(16,16))
plt.xticks(rotation=90)
plt.yticks(rotation=90)
sns.heatmap(corr, vmax=1.0, square=True,annot=True)

# 绘制各字段的直方图
import math
def plot_histogram(my_dataframe, cols = 5):
    rows = math.ceil(float(my_dataframe.shape[1]) / cols)
    fig = plt.figure(figsize=(20,15))
    for i, column in enumerate(my_dataframe.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if my_dataframe.dtypes[column] == np.object: # 类别属性
            my_dataframe[column].value_counts().plot(kind="bar", axes=ax)# 统计各种类数量
        else: # 值属性
            my_dataframe[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
plot_histogram(data)

#查看PURCHASES_INSTALLMENTS_FREQUENCY和PURCHASES_FREQUENCY的关系
plt.figure()
plt.scatter(data["PURCHASES_INSTALLMENTS_FREQUENCY"],data["PURCHASES_FREQUENCY"])
plt.xlabel("PURCHASES_INSTALLMENTS_FREQUENCY")
plt.ylabel("PURCHASES_FREQUENCY")

#删除部分字段
dropColumn=["ONEOFF_PURCHASES","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_TRX"]
data.drop(dropColumn,axis=1,inplace=True)

#查看类别变量的各个种类值
def show_labels(data):
    for i, column in enumerate(data.columns):
        if data.dtypes[column]==np.object:
            print("\n---"+column+"---\n")
            print(data[column].value_counts())
show_labels(data)

#PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
dataPca = pca.fit_transform(data)
explained_variance_ratio = pca.explained_variance_ratio_
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_), c='orange')
plt.xlabel('number of components')
plt.ylabel('cumulative explained varian')

#使用4个变量的PCA
pca = PCA(n_components = 4)
dataPca = pca.fit_transform(data)

#特征缩放
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X = scX.fit_transform(dataPca)

#使用肘部法则选择最优的K值
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter=300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
wcss

#绘制 K-WCSS折线图
plt.figure()
plt.plot(range(1, 20), wcss, 'ro-')
plt.title('ans')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.show()#k=5的时候下降不再明显

# 计算不同k值对应的轮廓系数
from sklearn.metrics import silhouette_score
for i in range(2, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter=300, random_state = 0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    silhouette = silhouette_score(X, y_kmeans)
    print('当聚类个数是%d时，对应的轮廓系数是%.4f' %(i, silhouette))

#选取k=7
kmeans = KMeans(n_clusters = 7, init = 'k-means++', n_init=10, max_iter=300, random_state = 0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
y_kmeans

# 可视化结果
plt.figure(figsize=(16,16))
ax = plt.subplot( projection='3d')  # 创建一个三维的绘图工程
ax.set_title('ans')  # 设置本图名称
ax.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],X[y_kmeans==0,2],c="blue")
ax.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],X[y_kmeans==1,2],c="cyan")
ax.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],X[y_kmeans==2,2],c="green")
ax.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],X[y_kmeans==3,2],c="black")
ax.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],X[y_kmeans==4,2],c="magenta")
ax.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],X[y_kmeans==5,2],c="tan")
ax.scatter(X[y_kmeans==6,0],X[y_kmeans==6,1],X[y_kmeans==6,2],c="darkgreen")
plt.show()

# 合并聚类结果
data["cluster"]=y_kmeans
data.cluster.value_counts()

#为每一个分组生成新的变量
cluster_0 = data[data['cluster'] == 0]
cluster_1 = data[data['cluster'] == 1]
cluster_2 = data[data['cluster'] == 2]
cluster_3 = data[data['cluster'] == 3]
cluster_4 = data[data['cluster'] == 4]
cluster_5 = data[data['cluster'] == 5]
cluster_6 = data[data['cluster'] == 6]

# 分析信用卡额度字段
credit_limit_df = pd.DataFrame(data={
'cluster_0':cluster_0.CREDIT_LIMIT.describe(),
'cluster_1':cluster_1.CREDIT_LIMIT.describe(),
'cluster_2':cluster_2.CREDIT_LIMIT.describe(),
'cluster_3':cluster_3.CREDIT_LIMIT.describe(),
'cluster_4':cluster_4.CREDIT_LIMIT.describe(),
'cluster_5':cluster_5.CREDIT_LIMIT.describe(),
'cluster_6':cluster_6.CREDIT_LIMIT.describe()})

#分析购买字段
purchases_df = pd.DataFrame(data={
'cluster_0':cluster_0.PURCHASES.describe(),
'cluster_1':cluster_1.PURCHASES.describe(),
'cluster_2':cluster_2.PURCHASES.describe(),
'cluster_3':cluster_3.PURCHASES.describe(),
'cluster_4':cluster_4.PURCHASES.describe(),
'cluster_5':cluster_5.PURCHASES.describe(),
'cluster_6':cluster_6.PURCHASES.describe()})

#对第四分组进行筛选
##挑选出频繁使用信用卡的用户
cluster_4.drop(cluster_4[cluster_4["BALANCE_FREQUENCY"]<0.8].index,axis=0,inplace=True)
##挑选出全款还款比例搞的用户
cluster_4.drop(cluster_4[cluster_4["PRC_FULL_PAYMENT"]<0.8].index,axis=0,inplace=True)






