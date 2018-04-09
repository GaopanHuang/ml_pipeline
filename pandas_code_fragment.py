#coding:utf-8
import numpy as np
import pandas as pd
import random


#####读取json或csv文件
df = pd.read_json('data.json')
df = pd.read_csv('data.csv', sep=', ', header=None) #注意csv文件读取的分隔符和是否有列名

#读取指定的列
df = pd.read_csv('data.csv', usecols=lambda x: x.upper() in ['A','B'])
df = pd.read_csv('data.csv', usecols=[1,2])#如果有列名，则直接in ['col1','col2']

#读取或剔除指定的行
df = pd.read_csv('data.csv', sep=', ', header=None, nrows=100)#读取100行
df = pd.read_csv('data.csv', sep=', ', header=None, skiprows=lambda x: x % 2 != 0)#剔除奇数行
df = pd.read_csv('data.csv', sep=', ', header=None, skiprows=100)#剔除前100行

#读取时对文件特定值标注为缺失值，如只把na字符标注为缺失值
df = pd.read_csv('data.csv',na_values=["na"])

#对于df直接to_csv保存的文件，header最前面会有个“,”，则读取如下：
df = pd.read_csv('data.csv',header=0,index_col=0)


#####保存csv或json文件
df.to_json('data1.json')
df.to_csv('data2.csv')

#保存header, 不保留index
df.to_csv('data2.csv',header=['c1','c2','c3'], index=None)
#不保存header
df.to_csv('data2.csv',header=None)
#不保存index
df.to_csv('data2.csv',index=None)
#不保存header和index
df.to_csv('data2.csv',header=None,index=None,mode='a+',)


#####例子数据
features = np.random.randint(0,10,size=[10,8])
target = np.random.randint(0,2,size=10)
df = pd.DataFrame(features)
df['target'] = target

#####df数据的获取或定位
#获取df的numpy的值
df.values

#获取指定的行，通过切片获取
df[1:3]
df[0:1]
df.loc[3]
df.loc[1:2]  #获取df的第1和2行，包括上限值
#获取指定的列
df['col1']
df.col1
df[2]
df.loc[:,1:3]
df.loc[:,['col1,'col2']]
#获取指定位置的值
df.loc[2:4,[0,2]]
df.loc[3,[0,2]]
df.loc[2,0]

#也可通过索引index获取行、列、指定位置的值
df.iloc[2]  #index=2的一行
df.iloc[:,1:3]  #第1,2列的值


####################找出最大值或最小值所在的位置
df.idxmax()       #每列最大值所在的位置
df.idxmax(axis=1) #每行最大值所在的位置


###########df数据的过滤
#过滤某些值
df[df['2']>0]  #获取第二列特征大于0的样本
df2[df2['E'].isin(['two','four'])]  #获取E列特征属于['two','four']的样本
#从df的所有10个样本中随机选取6个样本
take_out_set = df.ix[random.sample(df.index, 4)]
training_set = df[~(df.isin(take_out_set)).all(axis=1)]
#也可指定过滤哪些样本，如去除index为2,3,4的样本
i = [2,3,4]
take_out_set = df.ix[i]
training_set = df[~(df.isin(take_out_set)).all(axis=1)]
#过滤特征缺失的样本
df[df.col1.isnull()==True]

#判断特征缺失的数量
df.col1.isnull().sum()  #某列
df.isnull().sum()       #每列缺失的数量
df.isnull().sum().sum() #所有缺失的数量
#删除包含缺失特征的样本
df1.dropna(how='any')

#将缺失特征填充为指定值，前提是存在缺失值
df1.fillna(value=0)
#通过字典调用fillna, 实现不同的列填充不同的值
df.fillna({1: 0.5, 3: -1})

#####df数据的统计功能
df.count() #非空元素计算
df.min() #最小值
df.max() #最大值
df.idxmin() #最小值的位置，类似于R中的which.min函数
df.idxmax() #最大值的位置，类似于R中的which.max函数
df.quantile(0.25) #下4分位数，箱式图：Q1-3IQR, Q1-1.5IQR, Q1, Q2, Q3, Q3+1.5IQR, Q3+3IQR, IQR=Q3-Q1为四分位数差
df.quantile(0.75) #上4分位数
df.quantile(0.5) #中位数
df.median() #中位数
df.sum() #求和
df.mean() #均值
df.mode() #众数，出现最多次数的数
df.var() #方差
df.std() #标准差
df.mad() #平均绝对偏差
df.skew() #偏度
df.kurt() #峰度
df.describe() #一次性输出多个描述性统计指标,必须注意的是，descirbe方法只能针对序列或数据框，一维数组是没有这个方法的


#####数组拼接
#添加行，列名需要一样，否则会新增列，其他值通过NaN填充
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = pd.DataFrame(np.random.randn(1, 4), columns=['A','B','C','D'])
df.append(s, ignore_index=True)
#可一次添加多行
s = pd.DataFrame(np.random.randn(2, 4), columns=['A','B','C','D'])
df.append(s, ignore_index=True)
#用concat效率比append要高
pd.concat([pd.DataFrame([i*i+2], columns=['A']) for i in range(5)], ignore_index=True)

#按行拼接
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = pd.DataFrame(np.random.randn(2, 4), columns=['A','B','C','D'])
df = pd.concat([df, s], ignore_index=True)

#按列拼接，不足的值通过NaN填充
df = pd.DataFrame(np.random.randn(4, 2), columns=['A','B'])
s = pd.DataFrame(np.random.randn(5, 2), columns=['C','D'])
df = pd.concat([df, s], axis=1)
 