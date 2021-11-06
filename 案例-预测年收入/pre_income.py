# -*- coding: utf-8 -*-
## pre_income.py
##预测年收入

#导入包
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math

#导入数据集
data=pd.read_csv("./adult_income.csv")

#fnlwgt是序号，删除该列
data.drop(["fnlwgt"],axis=1,inplace=True)

# 绘制各字段的直方图
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

#查看education 和 education.num的关系
plt.figure()
plt.scatter(data["education"],data["education.num"])

#删除education列
data.drop(["education"],axis=1,inplace=True)

#统计缺失数据
def calcNull(data):
    nullSum=data.isnull().sum()
    nullSum=nullSum.drop(nullSum[nullSum.values==0].index)
    return nullSum
missing_data=calcNull(data)

#查看类别变量的各个种类值
def show_labels(data):
    for i, column in enumerate(data.columns):
        if data.dtypes[column]==np.object:
            print("\n---"+column+"---\n")
            print(data[column].value_counts())
show_labels(data)

#将native.country的？设置为others
data["native.country"]=data["native.country"].replace("?","Others")

#将workclass中为？并且occupation中也为？的项的值改为Others
data.loc[(data['workclass'] == '?') & (data['occupation'] == '?')
         ,['workclass','occupation']]="Others"

#删掉其余有？的行
data.drop(data[data['occupation'] == '?'].index,axis=0,inplace=True)

#将类别数据转换成哑变量
data=pd.get_dummies(data,drop_first=True)

#分离自变量和因变量
X=data.iloc[:,:-1]
Y=data.iloc[:,-1]

#拆分数据集和训练集
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=12)

#检查正样本的占比
print("训练集中正样本占比 {:.2f}".format(y_train.sum()/y_train.shape[0]))
print("测试集中正样本占比 {:.2f}".format(y_test.sum()/y_test.shape[0]))

#特征缩放
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


#建立模型
##逻辑回归模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression( class_weight='balanced')

##KNN模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

##支持向量机
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', gamma='scale', C=1.0 )

##朴素贝叶斯模型
from sklearn.naive_bayes import GaussianNB
gs = GaussianNB()

##随机森林模型训练
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500,max_depth=9, max_features='auto', min_samples_leaf=5)

# 定义k-fold函数 进行交叉验证
from sklearn.model_selection import cross_val_score
def evaluation_model(model):
    ac=cross_val_score(estimator=model,X=x_train,y=y_train,scoring="accuracy",cv=10,n_jobs=6,verbose=1)
    #print("%s evalustion: accuracy=%.4f,[std=%.4f] "%(model_name,ac.mean(),ac.std()))
    return ac
def print_result(model_name,ac):
    print("%s evalustion: accuracy=%.4f,[std=%.4f] "%(model_name,ac.mean(),ac.std()))


#执行模型并输出结果
lr_ac=evaluation_model(lr)
knn_ac=evaluation_model(knn)
svc_ac=evaluation_model(svc)
gs_ac=evaluation_model(gs)
rf_ac=evaluation_model(rf)

print_result("逻辑回归",lr_ac)
print_result("KNN",knn_ac)
print_result("支持向量机",svc_ac)
print_result("朴素贝叶斯",gs_ac)
print_result("随机森林",rf_ac)

#逻辑回归 evalustion: accuracy=0.8099,[std=0.0094] 
#KNN evalustion: accuracy=0.8276,[std=0.0057] 
#支持向量机 evalustion: accuracy=0.8488,[std=0.0057] 
#朴素贝叶斯 evalustion: accuracy=0.4434,[std=0.0393] 
#随机森林 evalustion: accuracy=0.8581,[std=0.0068] 

#模型调优，选取最优参数
from sklearn.model_selection import GridSearchCV
para=[{"criterion":["gini","entropy"],
       "n_estimators":[500,600],
       "max_depth":[15,16]}]
grid=GridSearchCV(estimator=rf,param_grid=para,scoring="accuracy",n_jobs=-1,cv=10,verbose=1)
grid=grid.fit(x_train,y_train)
best_ac = grid.best_score_ 
best_para = grid.best_params_ 
print('best_accuracy is: %.4f' %(best_ac))
print('best_parameters is: %s' %(best_para))
#best_accuracy is: 0.8619
#best_parameters is: {'criterion': 'gini', 'max_depth': 16, 'n_estimators': 500}

#使用最优模型和最优参数训练并预测
rf = RandomForestClassifier(n_estimators=500,max_depth=16, criterion="gini")
rf.fit(x_train,y_train)
y_pre=rf.predict(x_test)

#生成混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pre)
print("混淆矩阵:",cm)
print("准确度:",(cm[0][0]+cm[1][1])/cm.sum())

#获取属性的重要度排序
names=list(X)
rst=zip(map(lambda x:round(x,4),rf.feature_importances_),names)
rst=sorted(rst,reverse=True)








