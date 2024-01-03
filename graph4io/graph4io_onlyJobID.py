# AIIO Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy) and Ohio State
# University. All rights reserved.
import traceback

from utils import *
from UtilGetFeaturesInfos import *
import time
import random


file_tot_performace_tagged="/home/hungphd/media/aiio/data/sample_train.csv"
fpPreprocessFile="/home/hungphd/media/aiio/data/sample_train_small.csv"

# lstSubsetFeaturesPositive=list(range(0,45))
# lstSubsetFeaturesPositive.remove(2)
# lstSubsetFeaturesPositive.remove(3)
# lstSubsetFeaturesPositive.remove(4)
# lstSubsetFeaturesPositive.remove(5)
# lstSubsetFeaturesPositive.remove(6)
# lstSubsetFeaturesPositive.remove(1)
# lstSubsetFeaturesPositive.remove(11)
# lstSubsetFeaturesPositive.remove(12)
lstSubsetFeaturesPositive=[9,12,13]

time_str=time.strftime("%Y%m%d-%H%M%S")
fopResult='/home/hungphd/media/aiio/results/lightGBM_subset/'
fopCsvGNNTrain='./csvGraph4IO_big_train/'
fopCsvGNNTest='./csvGraph4IO_big_test/'
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

numberSelected=10000
numberTrain=7500
numberTest=2500
numDimension=2

# numberSelected=1000001
# numberTrain=750000
# numberTest=250000
cacheSize=2000

fpStat=fopResult+'stat.txt'
f1=open(fpStat,'r')
arrLines=f1.read().split('\n')
f1.close()
lstHeaderCols=[]
lstHeaderIndexes=[]
lstFullHeaderCols=[]
for i in range(1,len(arrLines)-1):
    # colName=arrLines[i].split('\t')[0]
    colName = 'col{}'.format(i)
    realIndex=i-1
    lstFullHeaderCols.append(colName)
    if realIndex in lstSubsetFeaturesPositive:
        lstHeaderCols.append(colName)
        lstHeaderIndexes.append(realIndex)

dictEdges={}
for i in range(0,len(lstHeaderCols)-1):
    # colCurrent=lstHeaderCols[i]
    # colNext=lstHeaderCols[i+1]
    # nameEdge='{}_AB_{}'.format(colCurrent,colNext)
    colCurrent = lstHeaderCols[i]
    colNext = lstHeaderCols[i+1]
    nameEdge = 'edge-{}-{}'.format(colCurrent, colNext).replace('-col','')
    dictEdges[nameEdge]=(colCurrent,nameEdge,colNext)
    nameEdgeReverse = 'edge-{}-{}'.format(colNext,colCurrent).replace('-col','')
    dictEdges[nameEdgeReverse] = (colNext, nameEdgeReverse, colCurrent)

lstYamlTrain=['dataset_name: {}\nedge_data:'.format('./csvGraph4IO_big_train')]
lstYamlTest=['dataset_name: {}\nedge_data:'.format('./csvGraph4IO_big_test')]

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

print(dictIndexEdges)
# input('bbb')
lstYamlTrain.append('node_data:')
lstYamlTest.append('node_data:')

for i in range(0,len(lstHeaderCols)):
    f1=open(fopCsvGNNTrain+'nodes_{}.csv'.format(i),'w')
    f1.write('graph_id,node_id,feat\n')
    f1.close()
    f1 = open(fopCsvGNNTest + 'nodes_{}.csv'.format(i),'w')
    f1.write('graph_id,node_id,feat\n')
    f1.close()
    strNode='- file_name: nodes_{}.csv\n  ntype: {}'.format(i,lstHeaderCols[i])
    lstYamlTrain.append(strNode)
    lstYamlTest.append(strNode)

strGraphInfo='graph_data:\n  file_name: graphs.csv'
lstYamlTrain.append(strGraphInfo)
lstYamlTest.append(strGraphInfo)

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
                continue
                pass
            elif idxLine<=numberTrain:
                idxGraph=idxLine-1
                lstItemValues=[float(it) for it in line.split(',')]
                labelItem=int(lstItemValues[len(lstItemValues)-1])

                lstStrGraphInfo.append('{},{}'.format(idxGraph,labelItem))
                idxNode=-1
                for idxFeat in range(0,len(lstHeaderIndexes)):
                    idxNode+=1

                    typeName=lstFullHeaderCols[lstHeaderIndexes[idxFeat]]
                    for idxFeat2 in range(0, len(lstHeaderIndexes)):
                        if idxFeat == idxFeat2:
                            # lstItemValues[lstHeaderIndexes[idxFeat]] = 0.1
                            strVectorInfo = '{},{}'.format(lstItemValues[lstHeaderIndexes[idxFeat]],
                                                           ','.join(
                                                               ['{}'.format(0.0)
                                                                for it in
                                                                range(0, numDimension-1)]))
                            # strVectorInfo = '{}'.format(lstItemValues[idxFeat])
                            strAddNode = '{},{},"{}"'.format(idxGraph, idxNode, strVectorInfo)
                            dictStrNodes[idxFeat].append(strAddNode)
                        else:
                            strVectorInfo = '{}'.format(','.join(
                                                               ['0.0'.format(lstItemValues[lstHeaderIndexes[idxFeat]])
                                                                for it in
                                                                range(0, numDimension)]))
                            # strVectorInfo = '{}'.format(lstItemValues[idxFeat])
                            strAddNode = '{},{},"{}"'.format(idxGraph, idxFeat2, strVectorInfo)
                            dictStrNodes[idxFeat].append(strAddNode)
                    if idxFeat>0:
                        prevTypeName=lstFullHeaderCols[lstHeaderIndexes[idxFeat-1]]
                        strKeyEdge = 'edge-{}-{}'.format(prevTypeName, typeName).replace('-col','')
                        idxEdge=dictIndexEdges[strKeyEdge]
                        prevNode = idxNode - 1
                        currentNode=idxNode
                        dictStrEdges[idxEdge].append('{},{},{}'.format(idxGraph,prevNode,currentNode))
                        strKeyEdgeReverse = 'edge-{}-{}'.format(typeName,prevTypeName).replace('-col','')
                        idxEdgeReverse = dictIndexEdges[strKeyEdgeReverse]
                        dictStrEdges[idxEdgeReverse].append('{},{},{}'.format(idxGraph, currentNode, prevNode))
                if idxLine==numberTrain:
                    isNeedToWrite=True
                pass
            else:
                idxGraph=idxLine-numberTrain-1
                isInTest=True
                isNeedToWrite=False
                lstItemValues = [float(it) for it in line.split(',')]
                labelItem = lstItemValues[len(lstItemValues) - 1]
                selectedFolder = fopCsvGNNTest
                lstStrGraphInfo.append('{},{}'.format(idxGraph, labelItem))
                idxNode=-1
                for idxFeat in range(0, len(lstHeaderIndexes)):
                    idxNode+=1
                    typeName = lstFullHeaderCols[lstHeaderIndexes[idxFeat]]
                    for idxFeat2 in range(0,len(lstHeaderIndexes)):
                        if idxFeat==idxFeat2:
                            # lstItemValues[lstHeaderIndexes[idxFeat]]=0.1
                            strVectorInfo = '{},{}'.format(lstItemValues[lstHeaderIndexes[idxFeat]],
                                                           ','.join(
                                                               ['{}'.format(0.0) for it in
                                                                range(0, numDimension-1)]))
                            # strVectorInfo = '{}'.format(lstItemValues[idxFeat])
                            strAddNode = '{},{},"{}"'.format(idxGraph, idxNode, strVectorInfo)
                            dictStrNodes[idxFeat].append(strAddNode)
                        else:
                            strVectorInfo = '{}'.format(','.join(
                                ['0.0'.format(lstItemValues[lstHeaderIndexes[idxFeat]])
                                 for it in
                                 range(0, numDimension)]))
                            # strVectorInfo = '{}'.format(lstItemValues[idxFeat])
                            strAddNode = '{},{},"{}"'.format(idxGraph, idxFeat2, strVectorInfo)
                            dictStrNodes[idxFeat].append(strAddNode)

                    if idxFeat > 0:
                        prevTypeName=lstFullHeaderCols[lstHeaderIndexes[idxFeat-1]]
                        strKeyEdge = 'edge-{}-{}'.format(prevTypeName, typeName).replace('-col','')
                        idxEdge = dictIndexEdges[strKeyEdge]
                        prevNode = idxNode - 1
                        currentNode = idxNode
                        dictStrEdges[idxEdge].append('{},{},{}'.format(idxGraph, prevNode, currentNode))
                        strKeyEdgeReverse =  'edge-{}-{}'.format(typeName,prevTypeName ).replace('-col','')
                        idxEdgeReverse = dictIndexEdges[strKeyEdgeReverse]
                        dictStrEdges[idxEdgeReverse].append('{},{},{}'.format(idxGraph, currentNode, prevNode))
                pass


        except Exception as e:
            traceback.print_exc()
        # if idxLine<=100:
        #     print(line.replace(',','\t'))
            # input('bbb')
        if idxLine % cacheSize == 0 or idxLine >= numberSelected or isNeedToWrite:
            print('{}\t{}'.format(idxLine,line))
            f1 = open(fpPreprocessFile, 'a')
            f1.write('\n'.join(lstOutputs) + '\n')
            f1.close()
            lstOutputs = []

            f1=open(selectedFolder+'graphs.csv','a')
            f1.write('\n'.join(lstStrGraphInfo)+'\n')
            f1.close()
            lstStrGraphInfo=[]

            for idxFeat in range(0, len(lstHeaderCols)):
                f1=open(selectedFolder+'nodes_{}.csv'.format(idxFeat),'a')
                f1.write('\n'.join(dictStrNodes[idxFeat])+'\n')
                f1.close()
                dictStrNodes[idxFeat]=[]

            idxEdge=-1
            for keyEdge in dictStrEdges.keys():
                lstEdgeValue=dictStrEdges[keyEdge]
                idxEdge+=1
                f1=open(selectedFolder+'edges_{}.csv'.format(idxEdge),'a')
                f1.write('\n'.join(dictStrEdges[idxEdge])+'\n')
                f1.close()
                dictStrEdges[keyEdge]=[]



        if idxLine>=numberSelected:
            break
        # print(line)
if len(lstOutputs)>0:
    f1 = open(fpPreprocessFile, 'a')
    f1.write('\n'.join(lstOutputs) + '\n')
    f1.close()
    lstOutputs = []
