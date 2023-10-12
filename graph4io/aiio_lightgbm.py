# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.
import traceback

from utils import *
import time



file_tot_performace_tagged="/home/hungphd/media/aiio/data/sample_train.csv"
fpPreprocessFile="/home/hungphd/media/aiio/data/sample_train_small.csv"

# lstSubsetFeaturesPositive=[37,40,29,33,38,30,34,28,31,35]
# lstSubsetFeaturesPositive=[37,40,29,33,38]
# lstSubsetFeaturesPositive=[25,24,14,16,8,19,21,20,43,44]
# lstSubsetFeaturesPositive=[25,24,14,16,8]
# lstSubsetFeaturesPositive=[37,40,29,33,38,30,34,28,31,35,25,24,14,16,8,19,21,20,43,44]
lstSubsetFeaturesPositive=list(range(0,45))
# lstSubsetFeaturesPositive.remove(0)
# lstSubsetFeaturesPositive.remove(2)
# lstSubsetFeaturesPositive.remove(3)
lstSubsetFeaturesPositive.remove(4)
lstSubsetFeaturesPositive.remove(5)
# lstSubsetFeaturesPositive.remove(22)
# lstSubsetFeaturesPositive.remove(6)
# lstSubsetFeaturesPositive.remove(18)

time_str=time.strftime("%Y%m%d-%H%M%S")
fopResult='/home/hungphd/media/aiio/results/lightGBM_subset/'
createDirIfNotExist(fopResult)

plot_result_file_name=fopResult+"io-ai-model-lightgbm-sparse-learning-curve-"+time_str+".pdf"
model_save_file_name=fopResult+"io-ai-model-lightgbm-sparse-"+time_str+".joblib"


print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)


from numpy import loadtxt
from matplotlib import pyplot
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from numpy import absolute
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from multiscorer import MultiScorer
from numpy import average
import joblib
import time
import scipy.sparse
from sklearn.metrics import *
from scipy.stats import *
## Set random seed
seed_value=48

import os
import random
os.environ['PYTHONHASHSEED']=str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

import lightgbm as lgb

numberSelected=1000001
cacheSize=2000
f1=open(file_tot_performace_tagged,'r')
lstOutputs=[]
idxLine=0
f1=open(fpPreprocessFile,'w')
f1.write('')
f1.close()
with open(file_tot_performace_tagged) as infile:
    for line in infile:
        try:
            idxLine+=1
            lstOutputs.append(line)

        except Exception as e:
            traceback.print_exc()
        # if idxLine<=100:
        #     print(line.replace(',','\t'))
            # input('bbb')
        if idxLine % cacheSize == 0 or idxLine == numberSelected:
            # print('{}\t{}'.format(idxLine,line))
            f1 = open(fpPreprocessFile, 'a')
            f1.write('\n'.join(lstOutputs) + '\n')
            f1.close()
            lstOutputs = []
        if idxLine==numberSelected:
            break
        # print(line)
if len(lstOutputs)>0:
    f1 = open(fpPreprocessFile, 'a')
    f1.write('\n'.join(lstOutputs) + '\n')
    f1.close()
    lstOutputs = []

# print('end file {}'.format(fpPreprocessFile))
# input('aaa')

# fpPreprocessFile=file_tot_performace_tagged

# load the train dataset
dataset = loadtxt(fpPreprocessFile, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
print(dataset.shape)
n_dims = dataset.shape[1]
# X = dataset[:,0:n_dims-1]
X = dataset[:,lstSubsetFeaturesPositive]

print("Before sparse.csr_matrix = ", type(X))
X=scipy.sparse.csr_matrix(X)
print("After  sparse.csr_matrix = ", type(X))

Y = dataset[:,n_dims-1]
print("max(Y) =", max(Y), ", min(Y) =", min(Y))
    
input_dim_size = n_dims -1
print("input_dim_size = ", input_dim_size)


#n_estimators=10000
model = lgb.LGBMRegressor(verbose=0,  n_estimators=100000, random_state=seed_value)
 


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
lenTrain=X_train.get_shape()[0]
lenSubsetTrain=X_test.get_shape()[0]
# X_train=X_train[:lenSubsetTrain*2]
# y_train=y_train[:lenSubsetTrain*2]
print("X_train.type=", type(X_train))
print("X_train.shape=", X_train.shape)
print("X_test.type=", type(X_test))
print("X_test.shape=", X_test.shape)
# input('aaaa')

lenColumns=X.get_shape()[1]
lstSpearmanRanks=[]
YList=Y.tolist()
for i in range(0,lenColumns):
    columnItem=X.getcol(i).toarray().tolist()
    columnItem=[it[0] for it in columnItem]
    # print('{}\t{}\t{}'.format(i+1,len(columnItem),len(YList)))
    spearmanScore,pValue=spearmanr(columnItem,YList)
    lstSpearmanRanks.append([spearmanScore,pValue])

lstValues=['{}\t{}\t{}'.format(i,lstSpearmanRanks[i][0],lstSpearmanRanks[i][1]) for  i in range(0,lenColumns)]
print('\n'.join(lstValues))
f1=open(fopResult+'spearmanr.txt','w')
f1.write('\n'.join(lstValues))
f1.close()







# define the datasets to evaluate each iteration
evalset = [(X_train, y_train), (X_test,y_test)]
# fit the model
model.fit(X_train, y_train,  eval_set=evalset, eval_metric='l1', early_stopping_rounds=10)

# evaluate performance
yhat = model.predict(X_test)
rmse_score = mean_squared_error(y_test, yhat, squared=False)
mae_score = mean_absolute_error(y_test, yhat)
r2Score=r2_score(y_test, yhat)
mape_score=mean_absolute_percentage_error(y_test, yhat)

print('RMSE\t{}'.format(rmse_score))
print('MAE\t{}'.format(mae_score))
print('R2\t{}'.format(r2Score))
print('MAPE\t{}'.format(mape_score))


lgb.plot_metric(model, xlabel='Iteration', ylabel='Loss', dataset_names=['valid_1'])
pyplot.savefig(plot_result_file_name)  
# pyplot.show()

#results = model.evals_result_
#pyplot.plot(results['validation_0']['rmse'], label='train')
#pyplot.xlabel('Iteration')
#pyplot.ylabel('Loss')
#pyplot.savefig(plot_result_file_name)  
#pyplot.show()

joblib.dump(model, model_save_file_name) 
print("plot_result_file_name =", plot_result_file_name)
print("model_save_file_name=", model_save_file_name)
#print(model_return)