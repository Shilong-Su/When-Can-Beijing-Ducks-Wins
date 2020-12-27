# coding:utf-8
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC,NuSVC,NuSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import roc_auc_score,roc_curve,mean_squared_error
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd

plt.rcParams['font.sans-serif']=['simsun']

def plot_ROC(t,s):
    fpr, tpr, thr = roc_curve(t, s, drop_intermediate=False)
    fpr, tpr = [0] + list(fpr), [0] + list(tpr)
    plt.plot(fpr, tpr)
    plt.show()


# 利用北京队和差值预测胜负
df = pd.read_excel('opponent_info（已清洗最终版）.xlsx')
#（差）数据与得分差
# df.drop(columns=['主场/客场','队伍','2分（京）','3分（京）','罚球（京）','进攻篮板（京）','防守篮板（京）','助攻（京）','犯规（京）','抢断（京）','失误（京）','盖帽（京）','扣篮（京）','被侵（京）','快攻（京）','得分（京）','得分（差）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','快攻（对)','得分（对）','时间','年','月','日','胜负','数字主场/客场','差值'],axis=1,inplace=True)
#（京）数据与得分差
df.drop(columns=['主场/客场','队伍','2分（差）','3分（差）','罚球（差）','进攻篮板（差）','防守篮板（差）','助攻（差）','犯规（差）','抢断（差）','失误（差）','盖帽（差）','扣篮（差）','被侵（差）','快攻（京）','得分（差）','得分（京）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','快攻（对)','得分（对）','时间','年','月','日','胜负','数字主场/客场','差值'],axis=1,inplace=True)
y = df['数字胜负']
df.drop(labels=['数字胜负'],axis=1,inplace=True)
x = df
x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=10)


# 遍历调参
param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
             'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
             'tol':[0.0001,0.005,0.00015,0.0002],
             'C': [1,0.5,1.5,2],
             'fit_intercept':[True,False],
             'max_iter':[50,100,150,200,250,300],
             'verbose':[0,1,2]}
print("Parameters:{}".format(param_grid))

grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5)
grid_search.fit(x_train,y_train)
print("Test set score:{:.2f}".format(grid_search.score(x_test,y_test)))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.2f}".format(grid_search.best_score_))

# （差）
# Model= LogisticRegression(C=1, fit_intercept= False, max_iter= 50, penalty= 'none', solver='newton-cg', tol=0.0001, verbose=0)
# （京）
Model= LogisticRegression(C=1, fit_intercept= True, max_iter= 50, penalty= 'none', solver='newton-cg', tol=0.0001, verbose=0)
Model.fit(x_train,y_train)
y_pred = Model.predict(x_test)

print(y_pred)
print(y_test)
k_fold = KFold(n_splits=3, shuffle=True)
scoring = 'accuracy'
score = cross_val_score(Model, x, y, cv=k_fold, n_jobs=2, scoring=scoring)
average_score = round(np.mean(score)*100, 2)

print('k_fold accuracy:', score)
print('average_score:', average_score)

# # 计算阈值
# df = pd.read_excel('opponent_info（已清洗最终版）.xlsx')
# y=df['数字胜负']
# #（京）数据与得分差
# df = df[['进攻篮板（京）','防守篮板（京）','助攻（京）','犯规（京）','抢断（京）','盖帽（京）','扣篮（京）']]
# #（差）数据与得分差
# # df = df[['进攻篮板（差）','防守篮板（差）','助攻（差）','犯规（差）','抢断（差）','盖帽（差）','扣篮（差）']]
# x = df
# x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=10)
#
#
# # 遍历调参
# param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
#              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
#              'tol':[0.0001,0.005,0.00015,0.0002],
#              'C': [1,0.5,1.5,2],
#              'fit_intercept':[True,False],
#              'max_iter':[50,100,150,200,250,300],
#              'verbose':[0,1,2]}
# print("Parameters:{}".format(param_grid))
#
# grid_search = GridSearchCV(LogisticRegression(),param_grid,cv=5)
# grid_search.fit(x_train,y_train)
# print("Test set score:{:.2f}".format(grid_search.score(x_test,y_test)))
# print("Best parameters:{}".format(grid_search.best_params_))
# print("Best score on train set:{:.2f}".format(grid_search.best_score_))
#
# # #此处可以换用多种分类方法
# # （京）
# Model= LogisticRegression(C=1.5, fit_intercept= True, max_iter= 150, penalty= 'l1', solver='liblinear', tol=0.005, verbose=0)
# # （差）
# # Model= LogisticRegression(C=1, fit_intercept= True, max_iter= 50, penalty= 'l1', solver='liblinear', tol=0.0001, verbose=0)
# Model.fit(x_train,y_train)
# y_pred = Model.predict(x_test)
#
# print(y_pred)
# print(y_test)
# k_fold = KFold(n_splits=3, shuffle=True)
# scoring = 'accuracy'
# score = cross_val_score(Model, x, y, cv=k_fold, n_jobs=2, scoring=scoring)
# average_score = round(np.mean(score)*100, 2)
#
# print('k_fold accuracy:', score)
# print('average_score:', average_score)
#
# x_test.head()
# x_test.describe()
# x_test_mean=[11.5,28.8,16.5,23.5,9.0,3.3,2.2]
#
# x_test_进攻篮板 = pd.DataFrame({'进攻篮板（京）': np.array(list(range(25)), dtype='int32'),
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': 16.5,
#                                '犯规（京）': 23.5,
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': 2.2})
# x1 = Model.predict_proba(x_test_进攻篮板)
# x1 = pd.DataFrame(x1,index = list(range(25)))
# print(x1)

# x_test_防守篮板 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': np.array(list(range(50)), dtype='int32'),
#                                '助攻（京）': 16.5,
#                                '犯规（京）': 23.5,
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': 2.2})
#
#
# x1 = Model.predict_proba(x_test_防守篮板)
# x1 = pd.DataFrame(x1,index = list(range(50)))
# print(x1)
#
# x_test_助攻 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': np.array(list(range(20)), dtype='int32'),
#                                '犯规（京）': 23.5,
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': 2.2})
#
#
# x1 = Model.predict_proba(x_test_助攻)
# x1 = pd.DataFrame(x1,index = list(range(20)))
# print(x1)
#
# x_test_犯规 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': 16.5,
#                                '犯规（京）': np.array(list(range(40)), dtype='int32'),
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': 2.2})
#
#
# x1 = Model.predict_proba(x_test_犯规)
# x1 = pd.DataFrame(x1,index = list(range(40)))
# print(x1)
#
# x_test_抢断 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': 16.5,
#                                '犯规（京）': 23.5,
#                                '抢断（京）': np.array(list(range(15)), dtype='int32'),
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': 2.2})
#
#
# x1 = Model.predict_proba(x_test_抢断)
# x1 = pd.DataFrame(x1,index = list(range(15)))
# print(x1)
#
# x_test_盖帽 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': 16.5,
#                                '犯规（京）': 23.5,
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': np.array(list(range(20)), dtype='int32'),
#                                '扣篮（京）': 2.2})
#
#
# x1 = Model.predict_proba(x_test_盖帽)
# x1 = pd.DataFrame(x1,index = list(range(20)))
# print(x1)
#
#
# x_test_扣篮 = pd.DataFrame({'进攻篮板（京）': 11.5,
#                                '防守篮板（京）': 28.8,
#                                '助攻（京）': 16.5,
#                                '犯规（京）': 23.5,
#                                '抢断（京）': 9.0,
#                                '盖帽（京）': 3.3,
#                                '扣篮（京）': np.array(list(range(10)), dtype='int32')})
#
#
# x1 = Model.predict_proba(x_test_扣篮)
# x1 = pd.DataFrame(x1,index = list(range(10)))
# print(x1)

# ##回归算法(得分)
# df = pd.read_excel('opponent_info（已清洗最终版）.xlsx')
# ##（京）数据与得分差
# df.drop(columns=['主场/客场','主场/客场.1','数字主场/客场','队伍','2分（差）','3分（差）','罚球（差）','进攻篮板（差）','防守篮板（差）','助攻（差）','犯规（差）','抢断（差）','失误（差）','扣篮（差）','被侵（差）','快攻（京）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','扣篮（差）','快攻（对)','得分（对）','差值','时间','年','月','日','盖帽（差）','胜负','数字胜负','得分（京）'],axis=1,inplace=True)
# #（差）数据与得分差
# # df.drop(columns=['主场/客场','主场/客场.1','数字主场/客场','队伍','2分（京）','3分（京）','罚球（京）','进攻篮板（京）','防守篮板（京）','助攻（京）','犯规（京）','抢断（京）','失误（京）','盖帽（京）','扣篮（京）','被侵（京）','快攻（京）','得分（京）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','快攻（对)','得分（对）','差值','时间','年','月','日','胜负','盖帽（京）','胜负','数字胜负'],axis=1,inplace=True)
# y=df['得分（差）']
# df.drop(labels=['得分（差）'],axis=1,inplace=True)
# x = df
# x.to_excel('test.xlsx')
# exit()
# x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
# # #此处可以换用多种分类方法，已在前述引用
# # 以下是已调好的参数
# Model= LinearRegression(fit_intercept=True, normalize=False,copy_X=True, n_jobs=1)
# # GridSearchCV调参
# # clf=GridSearchCV(Model,param_grid={'fit_intercept':[True,False], 'normalize':[True,False],'copy_X':[True,False], 'n_jobs':range(1,100)})
# # clf.fit(x_train,y_train)
# # print(clf.best_params_)
# # exit()
# Model.fit(x_train,y_train)
# y_pred = Model.predict(x_test)
# print(y_pred)
# print(y_test)
# print(Model.coef_)
# print(Model.intercept_)
# rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
# print('rmse:', rmse)

# 画图部分
##数据按胜负分类
# df = pd.read_excel('opponent_info（已清洗最终版）.xlsx')
# win = df.where(df['数字胜负']==1)
# lose = df.where(df['数字胜负']==0)
##画kdeplot折线图（可更换（京）与（差））
# fig, ax = plt.subplots()
# sn.kdeplot(win['扣篮（京）'],ax=ax, label='win')
# sn.kdeplot(lose['扣篮（京）'],ax=ax,label='lose')
# plt.legend(loc='upper right')
# plt.title('扣篮（京）')
# plt.show()
# exit()

#画2分3分等高图（可变（京）与（差））
# sn.kdeplot(lose['3分（差）'],lose['2分（差）'],color='yellow', shade=True,label='lose(yellow)')
# sn.kdeplot(win['3分（差）'],win['2分（差）'],color='blue', shade=True,label='win(blue)')
# plt.title('2分-3分等高图（差）')
# plt.legend(loc='upper right')
# plt.show()
# exit()


# correlation matrix
# df = pd.read_excel('opponent_info（已清洗最终版）.xlsx')
# # （京）数据与得分差
# df.drop(columns=['主场/客场','主场/客场.1','数字主场/客场','队伍','2分（差）','3分（差）','罚球（差）','进攻篮板（差）','防守篮板（差）','助攻（差）','犯规（差）','抢断（差）','失误（差）','扣篮（差）','被侵（差）','快攻（京）','得分（差）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','扣篮（差）','快攻（对)','得分（对）','差值','时间','年','月','日','盖帽（差）'],axis=1,inplace=True)
# #（差）数据与得分差
# df.drop(columns=['主场/客场','主场/客场.1','数字主场/客场','队伍','2分（京）','3分（京）','罚球（京）','进攻篮板（京）','防守篮板（京）','助攻（京）','犯规（京）','抢断（京）','失误（京）','盖帽（京）','扣篮（京）','被侵（京）','快攻（京）','得分（京）','主场/客场.1','队伍.1','2分（对）','3分（对）','罚球（对）','进攻篮板（对）','防守篮板（对）','助攻（对）','犯规（对）','抢断（对）','失误（对）','盖帽（对）','扣篮（对）','被侵（对）','快攻（对)','得分（对）','差值','时间','年','月','日'],axis=1,inplace=True)
# CorrMatrix = df.corr()
# fig = plt.figure('correlation matrix')
# sn.heatmap(CorrMatrix, annot=True, annot_kws={"size": 10}, fmt='.3f')
# plt.show()














