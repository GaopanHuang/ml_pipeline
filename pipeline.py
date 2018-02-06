#coding:utf-8
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

##################1.用pandas对数据进行初步的了解########################
df = pd.read_csv('train.csv', sep=',', header=None)
df.describe()
#通过箱线图分析特征分布情况，排除异常特征或样本
#箱线图：Q1-3IQR, Q1-1.5IQR, Q1, Q2, Q3, Q3+1.5IQR, Q3+3IQR
#IQR=Q3-Q1，四分位数差，如下画箱线图，有几行则画几个box，所以要转置T
plt.boxplot(df.iloc[:,1:10].T, showmeans=True), plt.show()



##################2.处理缺失特征########################################
df.col1 = df.col1.fillna(df.col1.mode()[0]) #用众数填充缺失值
df.col1 = df.col1.fillna(df.col1.mean()) #用均值填充缺失值

#用无缺失的数据建立模型来预测缺失数据的可能取值
imputer = KNeighborsRegressor()
df_nonnull = df[df.col1.isnull()==False]  #非特征缺失样本
df_null = df[df.col1.isnull()==True]      #特征缺失样本
cols = ['col2', 'col3', 'col4'] #利用其它三列特征预测缺失特征
imputer.fit(df_nonnull[cols], df_nonnull.col1)
train_values = imputer.predict(df_null[cols])
df_null.ix[:,'col1'] = train_values
new_df = df_nonnull.append(df_null)



##################3.图像数据增强，通过旋转、翻转、形变等###################
df += np.random.normal(0, 0.1, df.shape) #加gauss noise
df = df[::-1]                            #上下翻转
df = df.T[::-1].T                        #左右翻转
###旋转，假设旋转目标为df，大小为n*n
#构建反对角矩阵
back_diag = np.zeros(df.shape)
for i in range(len(df)):
  back_diag[i][(len(df))-1-i] = 1
np.dot(df,back_diag).T                   #逆时针旋转90度
np.dot(back_diag,np.dot(df,back_diag))   #逆时针旋转180度
np.dot(df.T,back_diag)                   #将矩阵逆时针旋转270度



##################4.选择基准模型，利用交叉验证CV##########################
#交叉验证可进行多次，通过多次的平均值估计模型性能
clf = xgb.XGBClassifier()
scoring = ['neg_log_loss', 'accuracy']
scores = cross_validate(clf, x_train, y_train, scoring=scoring, cv=10, return_train_score=True)
trainacc = scores['train_accuracy']
trainlogloss = -1*scores['train_neg_log_loss']
testacc = scores['test_accuracy']
testlogloss = -1*scores['test_neg_log_loss']
print 'train-acc: %f' % pd.DataFrame(trainacc).mean()
print 'train-logloss: %f' % pd.DataFrame(trainlogloss).mean()
print 'test-acc: %f' % pd.DataFrame(testacc).mean()
print 'test-logloss: %f' % pd.DataFrame(testlogloss).mean()



#######5,6反复进行，保证单模型能取得较好的结果，再去做模型集成#############
##################5.特征选择###############################################
##注意特征选择过程需要避免验证或测试样本参与特征选择过程
##transform适用于未参与模型拟合过程的样本进行特征选择
####去除低方差特征
sel = VarianceThreshold(threshold=2.5)
new_x_train = sel.fit_transform(x_train)

####五种单变量特征选择方法，可以统一用GenericUnivariateSelect通过不同参数实现
#需要注意的是x_train不能为负，可以通过加一个特定的值，得到特征后减去统一的值
new_x_train = SelectKBest(chi2, k=100).fit_transform(x_train, y_train)#卡方检验最好的前100个特征
new_x_train = GenericUnivariateSelect(score_func=chi2, mode='k_best', param=100)

####利用评估器的feature_importances_或coef_属性进行特征选择
selector = RFECV(estimator=SVC(),step=1,cv=StratifiedKFold(5),scoring='accuracy')
selector = selector.fit(x_train, y_train)
new_x_train = selector.transform(x_train)
#也可直接获取特征：
new_x_train = selector.fit_transform(x_train, y_train)
##lightgbm特征选择
clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for i in range(50):
  print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
selectedf_index = (np.argwhere((importances>0.0)==True)).reshape(-1,)#选择大于0的特征对应的索引
###上面的lightgbm也可以用RFECV来选择特征
selector = RFECV(estimator=clf,step=1,cv=StratifiedKFold(5))
new_x_train = selector.fit_transform(x_train)

####sklearn还有一种根据评估器来选择特征的通用方法SelectFromModel
model = SelectFromModel(clf, prefit=True)#对已完成训练的模型进行特征选择
new_x_train = model.transform(x_train)
new_x_test = model.transform(x_test)#该过程适用于对未参与拟合的测试样本进行特征选择
#对没有训练的模型，则需要先fit
model = SelectFromModel(clf, threshold=None, prefit=False, norm_order=1)
new_x_train = model.fit_transform(x_train, y_train)

#####特征与标签的相关性评估，亦可用于挑选特征
#pearson系数分析线性相关性，无需均值为0
import scipy.stats as stats
feat = np.array([0.3,0.6,0.1,0.9])
rst = np.array([0.2,0.9,0.6,0.2])
pear=stats.pearsonr(feat,rst)
print pear[0]
#或者直接利用DataFrame的corr方法实现
df.corr()#输出的结果即是不同列之间的pearson的值
df.corr('pearson')#因corr方法默认参数为pearson

#spearman系数作为变量之间单调联系强弱的度量，为等级相关系数
#与具体值无关，可利用DataFrame.corr方法直接计算
df.corr('spearman')#输出的结果即是不同列之间的spearman的值

#kendall相关系数也是等级相关系数，强调两列值对应大小关系的相同性，也叫和谐系数
#与具体值无关，可利用DataFrame.corr方法直接计算
df.corr('kendall')#输出的结果即是不同列之间的kendall的值

#互信息估计，从信息论的角度提出两随机变量的相关性
from sklearn.metrics import mutual_info_score as mi  
from sklearn.metrics import normalized_mutual_info_score as nmi  
mi(a,b)
nmi(a,b)

#MIC系数可度量较多相关性，包括三角函数关系等相关性，但结果不能反映具体哪种相关关系
from minepy import MINE
x = np.linspace(0, 1, 1000)
y = np.sin(10 * np.pi * x) + x
mine = MINE(alpha=0.6, c=15)
mine.compute_score(x, y)
print mine.mic()



##########################6.多个不同类型的模型调参，训练结果可视化，利用网格搜索GridSearchCV#############
#逐个参数进行调优，交叉反复多轮进行调优，先粗调后精调
clf = GradientBoostingClassifier()
param_grid = {'n_estimators': [40,80,100,200,500],
              #'loss': ['deviance', 'exponential'],
              #'learning_rate': [0.1,0.15],
              #'max_depth': [3, 5],
              }
grid_search = GridSearchCV(clf, param_grid=param_grid,scoring='neg_log_loss',cv=5,return_train_score=True)
grid_search.fit(X_train, y_train)

test_means = -1*grid_search.cv_results_['mean_test_score']
train_means = -1*grid_search.cv_results_['mean_train_score']
plt.plot(param_grid['n_estimators'],train_means,'b*-',label='train_logloss')
plt.plot(param_grid['n_estimators'],test_means,'r*-',label='test_logloss')
plt.legend(loc='upper right', fancybox=True)
plt.show()



##################7.模型集成stacking，尽量模型相关性不一样########################################
#####stacking集成模型代码
##stage 1
def get_singlemodel(clf, x_train, y_train, x_test, nFolds=5):
  s1_train = np.zeros((len(y_train),))
  s1_test = np.zeros((len(x_test),))
  s1_test_skf = np.empty((nFolds, len(x_test)))

  skf = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=7)
  onehot = OneHotEncoder(n_values=2, sparse=False)
  single_acc = 0.0
  single_logloss = 0.0
  for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
    x_train_skf = x_train.iloc[train_index]
    y_train_skf = y_train.iloc[train_index]
    x_val_skf = x_train.iloc[val_index]
    y_val_skf = y_train.iloc[val_index]

    clf.fit(x_train_skf, y_train_skf)
    y_val_pro_skf = clf.predict_proba(x_val_skf)[:,1]
    s1_train[val_index] = y_val_pro_skf
    
    y_val_pred_skf = y_val_pro_skf>0.5
    acc_pred_skf = np.mean(np.equal(y_val_skf, y_val_pred_skf))
    y_val_true = onehot.fit_transform(y_val_skf.values.reshape(-1,1))
    logloss = log_loss(y_val_true,y_val_pro_skf)
    print ('skfold %d: pred-acc:%f; logloss:%f' % (i,acc_pred_skf, logloss))
    single_acc += acc_pred_skf
    single_logloss += logloss

    s1_test_skf[i,:] = clf.predict_proba(x_test)[:,1]

  single_acc /= nFolds
  single_logloss /= nFolds
  print ('single model: acc:%f; logloss:%f' % (single_acc, single_logloss)) 
  s1_test = s1_test_skf.mean(axis=0)
  return s1_train.reshape(-1,1), s1_test.reshape(-1,1)

train = pd.read_csv('./data/train.csv', header=None)
x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
x_test = pd.read_csv('./data/test.csv', header=None)
rf = RandomForestClassifier()
gbdt = GradientBoostingClassifier()
rf_s1_train, rf_s1_test = get_singlemodel(clf=rf, x_train=x_train, y_train=y_train, x_test = x_test, nFolds=5)
gbdt_s1_train, gbdt_s1_test = get_singlemodel(clf=gbdt, x_train=x_train, y_train=y_train, x_test = x_test, nFolds=5)
x_train_s1 = np.concatenate(( rf_s1_train, gbdt_s1_train), axis=1)
x_test_s1 = np.concatenate(( rf_s1_test, gbdt_s1_test), axis=1)

##stage 2
clf_s2 = AdaBoostClassifier().fit(x_train_s1, y_train)
y_test_pro_s2 = clf_s2.predict_proba(x_test_s1)[:,1]



##################8.模型保存和加载############################################################
import pickle
print 'save model'
with open('clf.pickle','wb') as fw:
  pickle.dump(clf,fw)

print 'load model'
with open('clf.pickle','rb') as fr:
    new_clf = pickle.load(fr)
