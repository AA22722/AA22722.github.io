#!/usr/bin/env python

# scikit-learn
from sklearn import datasets

iris = datasets.load_iris()
# print(iris)
# print(iris.data,type(iris.data))
# print(iris.target,type(iris.target))
# print(iris.target_names,type(iris.target_names))
# print(iris.feature_names,type(iris.feature_names))

#使用交叉验证，把数据集分位训练样本和测试样本集
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target,test_size=0.1)

#建立模型
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(xtrain)
pre = model.predict(xtest)
center = model.cluster_centers_
print("KMeans 预测值")
print(pre)
print("输入数据的真实值")
print(ytest)
print("聚类的中心点")
print(center)



