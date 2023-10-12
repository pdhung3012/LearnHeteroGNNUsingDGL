# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.
import traceback

from utils import *
from UtilGetFeaturesInfos import *
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
fopCsvGNNTrain='./csvGraph4IO/train/'
fopCsvGNNTest='./csvGraph4IO/test/'
createDirIfNotExist(fopCsvGNNTrain)
createDirIfNotExist(fopCsvGNNTest)


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
numberTrain=750000
numberTest=250000
cacheSize=2000

fpStat=fopResult+'stat.txt'
f1=open(fpStat,'r')
arrLines=f1.read().split('\n')
f1.close()
lstHeaderCols=[]
for i in range(1,len(arrLines)):
    colName=arrLines[i].split('\t')[0]
    lstHeaderCols.append(colName)

dictEdges={}
for i in range(0,len(lstHeaderCols)-1):
    colCurrent=lstHeaderCols[i]
    colNext=lstHeaderCols[i+1]
    nameEdge='{}_AA_{}'.format(colCurrent,colNext)
    dictEdges[nameEdge]=(colCurrent,nameEdge,colNext)

lstYamlTrain=['dataset_name: {}\nedge_data:\n'.format(fopCsvGNNTrain)]
lstYamlTest=['dataset_name: {}\nedge_data:\n'.format(fopCsvGNNTest)]

idxEdges=-1
dictIndexEdges={}
for key in dictEdges.keys():
    idxEdges+=1
    val=dictEdges[key]
    dictIndexEdges[key]=idxEdges
    f1 = open(fopCsvGNNTrain + 'edges_{}.csv'.format(idxEdges),'w')
    f1.write('graph_id,src_id,dst_id\n')
    f1.close()
    f1 = open(fopCsvGNNTest + 'edges_{}.csv'.format(idxEdges),'w')
    f1.write('graph_id,src_id,dst_id\n')
    f1.close()

    strEdge='- file_name: edges_{}.csv\n  etype: [{}, {}, {}]'.format(idxEdges,val[0],val[1],val[2])
    lstYamlTrain.append(strEdge)
    lstYamlTest.append(strEdge)

lstYamlTrain.append('node_data:')
lstYamlTest.append('node_data:')

for i in range(0,len(lstHeaderCols)-1):
    f1=open(fopCsvGNNTrain+'nodes_{}.csv'.format(i),'w')
    f1.write('graph_id,node_id,feat\n')
    f1.close()
    f1 = open(fopCsvGNNTest + 'nodes_{}.csv'.format(i),'w')
    f1.write('graph_id,node_id,feat\n')
    f1.close()
    strNode='- file_name: nodes_{}.csv\n  ntype: {}'.format(i,lstHeaderCols[i])
    lstYamlTrain.append(strNode)
    lstYamlTest.append(strNode)


f1=open(fopCsvGNNTrain+'meta.yaml','w')
f1.write('\n'.join(lstYamlTrain))
f1.close()
f1=open(fopCsvGNNTest+'meta.yaml','w')
f1.write('\n'.join(lstYamlTest))
f1.close()



f1=open(fopCsvGNNTrain+'graphs.csv','w')
f1.write('graph_id,label\n')
f1.close()
f1=open(fopCsvGNNTest+'graphs.csv','w')
f1.write('graph_id,label\n')
f1.close()





f1=open(file_tot_performace_tagged,'r')

lstOutputs=[]
idxLine=-1
f1=open(fpPreprocessFile,'w')
f1.write('')
f1.close()
lstStrGraphInfo=[]
dictStrEdges={}
dictStrNodes={}
for i in range(0,len(lstHeaderCols)):
    dictStrNodes[i]=[]
idxEdges=-1
for key in dictEdges.keys():
    idxEdges+=1
    val=dictEdges[key]
    dictStrEdges[idxEdges]=[]

isInTest=False
isNeedToWrite=False
with open(file_tot_performace_tagged) as infile:
    for line in infile:
        try:
            idxLine+=1
            lstOutputs.append(line)
            selectedFolder = fopCsvGNNTrain
            if idxLine==0:
                # lstHeaderCols=line.split(',')
                pass
            elif idxLine<=numberTrain:
                lstItemValues=[float(it) for it in line.split(',')]
                labelItem=lstItemValues[len(lstItemValues)-1]

                lstStrGraphInfo.append('{},{}'.format(idxLine,labelItem))
                for idxFeat in range(0,len(lstItemValues)):
                    typeName=lstHeaderCols[idxFeat]
                    strAddNode='{},{},"{}"'.format(idxLine,idxFeat,lstItemValues[idxFeat])
                    dictStrNodes[idxFeat].append(strAddNode)
                    if idxFeat>0:
                        prevTypeName=lstHeaderCols[idxFeat-1]
                        strKeyEdge='{}_AA_{}'.format(prevTypeName,typeName)
                        idxEdge=dictIndexEdges[strKeyEdge]
                        dictStrEdges[idxEdge].append('{},{},{}'.format(idxLine,0,0))
                if idxLine==numberTrain:
                    isNeedToWrite=True
                pass
            else:
                isInTest=True
                isNeedToWrite=False
                lstItemValues = [float(it) for it in line.split(',')]
                labelItem = lstItemValues[len(lstItemValues) - 1]
                selectedFolder = fopCsvGNNTest
                lstStrGraphInfo.append('{},{}'.format(idxLine, labelItem))
                for idxFeat in range(0, len(lstItemValues)):
                    typeName = lstHeaderCols[idxFeat]
                    strAddNode = '{},{},"{}"'.format(idxLine, idxFeat, lstItemValues[idxFeat])
                    dictStrNodes[idxFeat].append(strAddNode)
                    if idxFeat > 0:
                        prevTypeName = lstHeaderCols[idxFeat - 1]
                        strKeyEdge = '{}_AA_{}'.format(prevTypeName, typeName)
                        idxEdge = dictIndexEdges[strKeyEdge]
                        dictStrEdges[idxEdge].append('{},{},{}'.format(idxLine, 0, 0))
                pass


        except Exception as e:
            traceback.print_exc()
        # if idxLine<=100:
        #     print(line.replace(',','\t'))
            # input('bbb')
        if idxLine % cacheSize == 0 or (idxLine+1) == numberSelected or isNeedToWrite:
            print('{}\t{}'.format(idxLine,line))
            f1 = open(fpPreprocessFile, 'a')
            f1.write('\n'.join(lstOutputs) + '\n')
            f1.close()
            lstOutputs = []

            f1=open(selectedFolder+'graphs.csv','a')
            f1.write('\n'.join(lstStrGraphInfo))
            f1.close()
            lstStrGraphInfo=[]

            for idxFeat in range(0, len(lstHeaderCols)-1):
                f1=open(selectedFolder+'nodes_{}.csv'.format(idxFeat),'a')
                f1.write('\n'.join(dictStrNodes[idxFeat])+'\n')
                f1.close()
                dictStrNodes[idxFeat]=[]

            idxEdges=-1
            for keyEdge in dictStrEdges.keys():
                lstEdgeValue=dictStrEdges[keyEdge]
                idxEdges+=1
                f1=open(selectedFolder+'edges_{}.csv'.format(idxEdges),'a')
                f1.write('\n'.join(dictStrEdges[idxFeat])+'\n')
                f1.close()
                dictStrEdges[keyEdge]=[]



        if idxLine==numberSelected:
            break
        # print(line)
if len(lstOutputs)>0:
    f1 = open(fpPreprocessFile, 'a')
    f1.write('\n'.join(lstOutputs) + '\n')
    f1.close()
    lstOutputs = []
