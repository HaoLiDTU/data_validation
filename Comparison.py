
# -*- coding: utf-8 -*-
"""
Created on Sat April  9 23:55:19 2021

@author: Hao Li (haoli@dtu.dk)
"""

import pandas as pd
from scipy import interp
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBRegressor as XGBR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn import preprocessing
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from time import time
import datetime
import numpy as np
from sklearn.model_selection import KFold


font = {'family': 'Arial Unicode MS',
        'color':  'darkred',
        'weight': 'normal',
        'size': 8,
        }


seed=8181
"""
datatotal=pd.read_csv("Chemistry.csv",header=None)
X=datatotal.drop(29,axis=1)
X=preprocessing.scale(X)
y=np.array(datatotal.iloc[:,29])
X_train,X_test,Y_train,Y_test=TTS(X,y,test_size=0.1,random_state=seed)
"""

##18881
seed=18888
#1188

font = {'family': 'Arial Unicode MS',
        'color':  'darkred',
        'weight': 'normal',
        'size': 8,
        }

###extreme tree
datatotal=pd.read_csv("Chemistry_f.csv")
X=datatotal.drop("Y",axis=1)
X=preprocessing.scale(X)
y=datatotal.iloc[:,26]

importances=np.loadtxt("importance1.txt")
im=np.sort(importances)[::-1]
arg=np.argsort(importances)[::-1]


i=20
t=arg[:i]
X=X[:,np.array(t)]
X=preprocessing.scale(X)
y=datatotal.iloc[:,26]
X_train,X_test,Y_train,Y_test=TTS(X,y,test_size=0.2,random_state=seed)

x=np.arange(-1,2,step=0.001)
y=x


###Xgboost
fig = plt.figure(figsize=(8,16))
ax = fig.subplots(3,2)
reg=XGBR(silent=True,n_estimators=200,max_depth=3,learning_rate=0.26,reg_lambda=0.09).fit(X_train,Y_train)
xgb_pre=reg.predict(X_test)
xgb_pre_tr=reg.predict(X_train)
xgb_avg=CVS(reg,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
xgb_mse=metrics.mean_squared_error(xgb_pre,Y_test)
xgb_r2=metrics.explained_variance_score(xgb_pre,Y_test)
xgb_mae=metrics.mean_absolute_error(xgb_pre,Y_test)
print("xgb_r2",xgb_r2)
#print("xgb_mse",xgb_mse)
print("xgb_mae",xgb_mae)
#plt.subplot(121)
#plt.figure(figsize=(10,8))
#ax1.text(x=1.36,y=0,s="R^2=0.987",fontdict=font)
#ax[0,0].text(x=1.36,y=-0.3,s="MSE=0.0043",fontdict=font)
ax[0,0].text(x=1.36,y=-0.3,s="RMSE=0.075",fontdict=font)
ax[0,0].text(x=1.36,y=-0.6,s="MAE=0.052",fontdict=font)
ax[0,0].set_title("Xgboost")
ax[0,0].plot(x,y,linewidth=3.81)
ax[0,0].scatter(Y_train,xgb_pre_tr,color = 'blue', s = 15)
ax[0,0].scatter(Y_test,xgb_pre,color = 'red', s = 50)
#plt.show() 



###GBDT
gbdt=GradientBoostingRegressor(n_estimators=200,max_depth=7,learning_rate=0.07)
gbdt.fit(X_train,Y_train)
gbdt_avg=CVS(gbdt,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
gbdt_pre_tr=gbdt.predict(X_train)
gbdt_pre=gbdt.predict(X_test)
gbdt_mse=metrics.mean_squared_error(gbdt_pre,Y_test)
gbdt_r2=metrics.explained_variance_score(gbdt_pre,Y_test)
gbdt_mae=metrics.mean_absolute_error(gbdt_pre,Y_test)
#print("mse",gbdt_mse,"r2",gbdt_r2,"mae",gbdt_mae)
#plt.subplot(122)
#plt.figure(figsize=(10,8))
#ax2.text(x=1.36,y=0,s="R^2=0.982",fontdict=font)
#fig, ax2 = plt.subplots(3, 2, 2)
ax[0,1].text(x=1.36,y=-0.3,s="RMSE=0.075",fontdict=font)
ax[0,1].text(x=1.36,y=-0.6,s="MAE=0.066",fontdict=font)
ax[0,1].set_title("GBDT")
ax[0,1].plot(x,y,linewidth=3.81)
ax[0,1].scatter(Y_test,gbdt_pre,color = 'red', s = 56)
#ax[0,1].subplots_adjust(hspace=0.5)
#plt.show() 



###Adaboost

adb = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6, min_samples_split=10, min_samples_leaf=5),n_estimators=300,learning_rate=0.06)
adb.fit(X_train,Y_train)
adb_pre=adb.predict(X_test)
adb_avg=CVS(adb,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
adb_mse=metrics.mean_squared_error(adb_pre,Y_test)
adb_r2=metrics.explained_variance_score(adb_pre,Y_test)
adb_mae=metrics.mean_absolute_error(adb_pre,Y_test)
#plt.subplot(211)
#plt.figure(figsize=(3,8))
#ax3.text(x=1.36,y=0,s="R^2=0.984",fontdict=font)
#fig, ax3 = plt.subplots(3, 2, 3)
ax[1,0].text(x=1.36,y=-0.3,s="RMSE=0.083",fontdict=font)
ax[1,0].text(x=1.36,y=-0.6,s="MAE=0.061",fontdict=font)
ax[1,0].set_title("Adaboost")
ax[1,0].plot(x,y,linewidth=3.81)
ax[1,0].scatter(Y_test,adb_pre,color = 'red', s = 56)
#plt.show()


###Randomforest
rf = RandomForestRegressor(max_depth=10,min_samples_leaf=5,n_estimators=300,max_features = "auto")
rf.fit(X_train,Y_train)
rf_avg=CVS(rf,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
rf_pre=rf.predict(X_test)
rf_mse=metrics.mean_squared_error(rf_pre,Y_test)
rf_r2=metrics.explained_variance_score(rf_pre,Y_test)
rf_mae=metrics.mean_absolute_error(rf_pre,Y_test)
#ax2.subplot(211)
#ax4.text(x=1.36,y=0,s="R^2=0.971",fontdict=font)
#fig, ax4 = plt.subplots(3, 2, 4)
ax[1,1].text(x=1.36,y=-0.3,s="RMSE=0.109",fontdict=font)
ax[1,1].text(x=1.36,y=-0.6,s="MAE=0.088",fontdict=font)
ax[1,1].set_title("RandomForest")
ax[1,1].plot(x,y,linewidth=3.61)
ax[1,1].scatter(Y_test,rf_pre,color = 'red', s = 56)
plt.subplots_adjust(hspace=0.5)
#plt.show()


###SVR
#fig, (ax1, ax2) = plt.subplots(1, 2)
svr=SVR(kernel='rbf',C=281,gamma=.011)
svr.fit(X_train,Y_train)
svr_avg=CVS(svr,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
svr_pre=svr.predict(X_test)
svr_mse=metrics.mean_squared_error(svr_pre,Y_test)
svr_r2=metrics.explained_variance_score(svr_pre,Y_test)
svr_mae=metrics.mean_absolute_error(svr_pre,Y_test)
#plt.subplot(325)
#ax5.text(x=1.36,y=0,s="R^2=0.98",fontdict=font)
#fig, ax5 = plt.subplots(3, 2, 5)
ax[2,0].text(x=1.36,y=-0.3,s="RMSE=0.108",fontdict=font)
ax[2,0].text(x=1.36,y=-0.6,s="MAE=0.086",fontdict=font)
ax[2,0].set_title("SVR")
ax[2,0].plot(x,y,linewidth=3.68)
ax[2,0].scatter(Y_test,svr_pre,color = 'red', s = 56)
#plt.show()


###NN
MLP = MLPRegressor(solver='lbfgs', alpha=1e-10,hidden_layer_sizes=(33,12,1), random_state=1)
MLP.fit(X_train,Y_train)
MLP_avg=CVS(MLP,X_train,Y_train,scoring="neg_mean_absolute_error",cv=5).mean()
mlp_pre=MLP.predict(X_test)
mlp_mse=metrics.mean_squared_error(mlp_pre,Y_test)
mlp_r2=metrics.explained_variance_score(mlp_pre,Y_test)
mlp_mae=metrics.mean_absolute_error(mlp_pre,Y_test)
#print("mse",mlp_mse,"r2",mlp_r2,"mae",mlp_mae)
#plt.subplot(326)
#ax6.text(x=1.36,y=0,s="R^2=0.98",fontdict=font)
#fig, ax6 = plt.subplots(3, 2, 6)
ax[2,1].text(x=1.36,y=-0.3,s="RMSE=0.099",fontdict=font)
ax[2,1].text(x=1.36,y=-0.6,s="MAE=0.098",fontdict=font)
ax[2,1].set_title("Neural Network")
plt.plot(x,y,linewidth=3.68)
ax[2,1].scatter(Y_test,mlp_pre,color = 'red', s = 56)
plt.subplots_adjust(hspace=0.5)
plt.tight_layout(pad=1.5)
plt.show()


print("xgb",xgb_mae,"CV",xgb_avg)
#print("xgb",xgb_r2)
print("gbdt",gbdt_mae,"CV",gbdt_avg)
#print("gbdt",gbdt_r2)
print("adb",adb_mae,"CV",adb_avg)
#print("adb",adb_r2)
print("rf",rf_mae,"CV",rf_avg)
#print("rf",rf_r2)
print("svr",svr_mae,"CV",svr_avg)
#print("svr",svr_r2)
print("mlp",mlp_mae,"CV",MLP_avg)
#print("mlp",mlp_r2)