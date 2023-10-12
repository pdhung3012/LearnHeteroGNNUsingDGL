import copy
import traceback
from statistics import *

from utils import *

def getFeatureStatistics(lstInputFeatures,numberScale):
    lstOutputFeatures=copy.copy(lstInputFeatures)
    lstOutputFeatures=sorted(lstOutputFeatures)
    lstStatistics=[]
    lstScaleByRange=[]
    setOutputFeatures = None

    try:
        minStat=min(lstOutputFeatures)
        maxStat=max(lstOutputFeatures)
        meanStat=mean(lstOutputFeatures)
        medianStat=median(lstOutputFeatures)
        setOutputFeatures=set(lstOutputFeatures)
        sizeUniqueValue=len(setOutputFeatures)
        lenFeatures=len(lstOutputFeatures)
        lstStatistics=[minStat,maxStat,meanStat,medianStat,sizeUniqueValue]
        offsetRank=lenFeatures//numberScale
        for i in range(0,numberScale):
            if i==0:
                lstScaleByRange.append(minStat)
            elif i==numberScale-1:
                lstScaleByRange.append(maxStat)
            else:
                itemScore= lstOutputFeatures[i*offsetRank]
                lstScaleByRange.append(itemScore)

    except Exception as e:
        traceback.print_exc()
    return lstOutputFeatures,setOutputFeatures,lstStatistics,lstScaleByRange
