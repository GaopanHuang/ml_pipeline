#coding:utf-8
import numpy as np
import pandas as pd
import random


#####��ȡjson��csv�ļ�
df = pd.read_json('data.json')
df = pd.read_csv('data.csv', sep=', ', header=None) #ע��csv�ļ���ȡ�ķָ������Ƿ�������

#��ȡָ������
df = pd.read_csv('data.csv', usecols=lambda x: x.upper() in ['A','B'])
df = pd.read_csv('data.csv', usecols=[1,2])#�������������ֱ��in ['col1','col2']

#��ȡ���޳�ָ������
df = pd.read_csv('data.csv', sep=', ', header=None, nrows=100)#��ȡ100��
df = pd.read_csv('data.csv', sep=', ', header=None, skiprows=lambda x: x % 2 != 0)#�޳�������
df = pd.read_csv('data.csv', sep=', ', header=None, skiprows=100)#�޳�ǰ100��

#��ȡʱ���ļ��ض�ֵ��עΪȱʧֵ����ֻ��na�ַ���עΪȱʧֵ
df = pd.read_csv('data.csv',na_values=["na"])

#����dfֱ��to_csv������ļ���header��ǰ����и���,�������ȡ���£�
df = pd.read_csv('data.csv',header=0,index_col=0)


#####����csv��json�ļ�
df.to_json('data1.json')
df.to_csv('data2.csv')

#����header, ������index
df.to_csv('data2.csv',header=['c1','c2','c3'], index=None)
#������header
df.to_csv('data2.csv',header=None)
#������index
df.to_csv('data2.csv',index=None)
#������header��index
df.to_csv('data2.csv',header=None,index=None,mode='a+',)


#####��������
features = np.random.randint(0,10,size=[10,8])
target = np.random.randint(0,2,size=10)
df = pd.DataFrame(features)
df['target'] = target

#####df���ݵĻ�ȡ��λ
#��ȡdf��numpy��ֵ
df.values

#��ȡָ�����У�ͨ����Ƭ��ȡ
df[1:3]
df[0:1]
df.loc[3]
df.loc[1:2]  #��ȡdf�ĵ�1��2�У���������ֵ
#��ȡָ������
df['col1']
df.col1
df[2]
df.loc[:,1:3]
df.loc[:,['col1,'col2']]
#��ȡָ��λ�õ�ֵ
df.loc[2:4,[0,2]]
df.loc[3,[0,2]]
df.loc[2,0]

#Ҳ��ͨ������index��ȡ�С��С�ָ��λ�õ�ֵ
df.iloc[2]  #index=2��һ��
df.iloc[:,1:3]  #��1,2�е�ֵ


####################�ҳ����ֵ����Сֵ���ڵ�λ��
df.idxmax()       #ÿ�����ֵ���ڵ�λ��
df.idxmax(axis=1) #ÿ�����ֵ���ڵ�λ��


###########df���ݵĹ���
#����ĳЩֵ
df[df['2']>0]  #��ȡ�ڶ�����������0������
df2[df2['E'].isin(['two','four'])]  #��ȡE����������['two','four']������
#��df������10�����������ѡȡ6������
take_out_set = df.ix[random.sample(df.index, 4)]
training_set = df[~(df.isin(take_out_set)).all(axis=1)]
#Ҳ��ָ��������Щ��������ȥ��indexΪ2,3,4������
i = [2,3,4]
take_out_set = df.ix[i]
training_set = df[~(df.isin(take_out_set)).all(axis=1)]
#��������ȱʧ������
df[df.col1.isnull()==True]

#�ж�����ȱʧ������
df.col1.isnull().sum()  #ĳ��
df.isnull().sum()       #ÿ��ȱʧ������
df.isnull().sum().sum() #����ȱʧ������
#ɾ������ȱʧ����������
df1.dropna(how='any')

#��ȱʧ�������Ϊָ��ֵ��ǰ���Ǵ���ȱʧֵ
df1.fillna(value=0)
#ͨ���ֵ����fillna, ʵ�ֲ�ͬ������䲻ͬ��ֵ
df.fillna({1: 0.5, 3: -1})

#####df���ݵ�ͳ�ƹ���
df.count() #�ǿ�Ԫ�ؼ���
df.min() #��Сֵ
df.max() #���ֵ
df.idxmin() #��Сֵ��λ�ã�������R�е�which.min����
df.idxmax() #���ֵ��λ�ã�������R�е�which.max����
df.quantile(0.25) #��4��λ������ʽͼ��Q1-3IQR, Q1-1.5IQR, Q1, Q2, Q3, Q3+1.5IQR, Q3+3IQR, IQR=Q3-Q1Ϊ�ķ�λ����
df.quantile(0.75) #��4��λ��
df.quantile(0.5) #��λ��
df.median() #��λ��
df.sum() #���
df.mean() #��ֵ
df.mode() #��������������������
df.var() #����
df.std() #��׼��
df.mad() #ƽ������ƫ��
df.skew() #ƫ��
df.kurt() #���
df.describe() #һ����������������ͳ��ָ��,����ע����ǣ�descirbe����ֻ��������л����ݿ�һά������û�����������


#####����ƴ��
#����У�������Ҫһ��������������У�����ֵͨ��NaN���
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = pd.DataFrame(np.random.randn(1, 4), columns=['A','B','C','D'])
df.append(s, ignore_index=True)
#��һ����Ӷ���
s = pd.DataFrame(np.random.randn(2, 4), columns=['A','B','C','D'])
df.append(s, ignore_index=True)
#��concatЧ�ʱ�appendҪ��
pd.concat([pd.DataFrame([i*i+2], columns=['A']) for i in range(5)], ignore_index=True)

#����ƴ��
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = pd.DataFrame(np.random.randn(2, 4), columns=['A','B','C','D'])
df = pd.concat([df, s], ignore_index=True)

#����ƴ�ӣ������ֵͨ��NaN���
df = pd.DataFrame(np.random.randn(4, 2), columns=['A','B'])
s = pd.DataFrame(np.random.randn(5, 2), columns=['C','D'])
df = pd.concat([df, s], axis=1)
 