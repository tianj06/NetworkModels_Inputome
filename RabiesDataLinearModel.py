# -*- coding: utf-8 -*-
"""
Import matlab formatted data 
Ju Tian 2015.5
"""
import scipy.io as sio
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import collections
from matplotlib.ticker import MaxNLocator


#%% preprocessing step 1: get subset of data by trial Type
#trialName = {'90%water':0, '50% reward':1, '10% reward':2, '80% airpuff':3,	
#             'omission 90% water':4, 'omission 50% reward':5,	'omssion 10% reward':6,
#             'omission airpuff':7,	'free reward':8,	'free airpuff':9}
def preprocessData(X,RemainIndex):
    """
    input X, array of psth, rows are units, columns are timestamps
    This function remove specific trialtypes data, fill missing values and 
    scale the data
    """  
    if len(X.shape) == 2:
        X = np.array(X)[RemainIndex,:]
    elif len(X.shape) == 1:
        X = np.array(X)[RemainIndex]
    # preprocessing step 2: fill in missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X = imp.fit_transform(X)
    
    # preprocessing step 3: fill scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_X = min_max_scaler.fit_transform(X)
    return normalized_X

#%% bootstrap the fitting error
def bootstrp_fitting_error(X,y,clf,n=50,N=500):
    train_error = list()
    min_train_error = 10000
    for j in range(N):   # N bootstrap number
        ind = random.sample(range(1, np.shape(X)[1]), n)
        X_sample = X[:,ind]
        clf.fit(X_sample,y)
        e = metrics.mean_squared_error(clf.predict(X_sample),y)    
        train_error.append(e)
        if e<min_train_error:
            min_train_error = e
            min_error_index = ind 
    return (train_error,min_error_index)
    

#%% plot the most frequent units for good fitting
def countTopFrequentUnits(List,N=12):
    Count = collections.Counter([item for sublist in List for item in sublist])
    sortedValue = sorted(Count.values(),reverse=True)
    result = np.zeros((N,2))
    i = 0
    for item in Count.items():
        if (item[1]>= sortedValue[N])&(i<N):
            result[i,0] = item[0]
            result[i,1] = float(item[1])/len(List)
            i+=1
    return result


def panelPSTHplotting(X,subtitles=[0],xtickLabels=[0]):
    """
    X is a list. Each element in X is a unit's PSTH, in the format of list of list 
    For an element in X, its dimension is n*m, n number of trialtypes, m trial
    response    
    """
    N = len(X)
    W = 3
    H = N/3
    colors = ((0.5, 0.5,0.5),(0,0,1),(0.12,0.57,1))
    f,axes = plt.subplots(W,H)
    f.subplots_adjust(hspace=.5)
    
    plotcount=0
    for i in range(W):
        for j in range(H):
            if plotcount<N:
                k=0
                for data in X[plotcount]:
                    axes[i,j].plot(data,color=colors[k])
                    axes[i,j].yaxis.set_major_locator(MaxNLocator(3))
                    k+=1
                if len(subtitles)==N:
                    axes[i,j].set_title(subtitles[plotcount])
                if len(xtickLabels)==N:
                    axes[i,j].set_xticklabels(xtickLabels[plotcount])
                plotcount+=1

    return (f,axes)

#%% plot the fitted psth given a specific number of inputs. (not useful)
def bootpsth(N=10):
    X_sample = All_input[:,random.sample(range(1, 124), N)]
    clf.fit(X_sample,DA_output)
    predicted_PSTH = clf.predict(X_sample)
    plt.plot(DA_output)
    plt.plot(predicted_PSTH)
    plt.ylabel('normalized response')
    plt.legend(['DA','predicted'])

#%% leave N out to look at the stability of weights

def weightStability(X,y,clf,N=5):    
    
    coefMatrix=np.full([np.shape(X)[1],100],np.nan,dtype='float')
    for i in range(100):
        remove_idx = random.sample(range(np.shape(X)[1]), N)
        ind = range(np.shape(X)[1])
        for j in remove_idx:
            ind.remove(j)
        X_sample = X[:,ind]
        clf.fit(X_sample,y)
        coefMatrix[ind,i]=clf.coef_
    meanCoef=np.nanmean(coefMatrix,axis=1)
    normalizedError = list()
    for i in range(100):
        NanIdx = np.isnan(coefMatrix[:,i])
        #ne = np.linalg.norm( coefMatrix[~NanIdx,i] - meanCoef[~NanIdx])/np.linalg.norm(meanCoef[~NanIdx])
        # the metrics used for stability, subject to change
        ne = np.corrcoef(coefMatrix[~NanIdx,i],meanCoef[~NanIdx])
        normalizedError.append(ne)
    return normalizedError

#%% since the weights looks pretty stable, now look at coefficients distribution
def weigthDistributionPlot(X,y,clf,brainAreaCode,GroupFactor):
    clf.fit(X,y)
    # plot fitting
    f,ax = plt.subplots(2)
    ax[0].plot(clf.predict(X))
    ax[0].plot(y)
    ax[0].legend(['DA','predicted'])
    ax[0].set_ylabel('Normalized firing rate')
    ax[0].set_title('N inputs = %d' %np.shape(X)[1])
    ax[0].set_xticks(range(0,300,50), minor=False)
    ax[0].xaxis.grid(True,linewidth=2)
    ax[0].yaxis.grid(None)
    ax[0].set_xticklabels([])

    
    coef = clf.coef_   
    sns.violinplot(pd.Series(coef),groupby=GroupFactor,inner='points',
                   names=brainAreaCode,ax=ax[1])
    ax[1].set_ylabel('Weight')
    ax[1].yaxis.set_major_locator(MaxNLocator(4))
    return(coef)

#%% 
def subsetPredictionPlot(neuronIdx,ax,X,Weights):
    subPredict= np.average(X[:,neuronIdx], axis=1,weights=Weights[neuronIdx])
    subPredict = subPredict*sum(Weights[neuronIdx])
    #plt.figure()
    ax.plot(DA_output*abs(np.sum(Weights[neuronIdx])))
    ax.plot(subPredict)
    ax.legend(['scaled DA','predicted'],bbox_to_anchor=(1.25, 1.1))
#    for i in range(6):
#        ax.text(25+i*50,0.8,trialtypeNames[i])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticks(range(0,300,50), minor=False)
    ax.xaxis.grid(True,linewidth=2)
    ax.yaxis.grid(None)
    return subPredict    

#%%
def crossPredictionStability(X,y,N=5):
    subIndex = range(100)+range(50*5,50*6)
    errorMatrix=np.zeros((100,2),dtype='float')
    for i in range(100):
        remove_idx = random.sample(range(1, np.shape(X)[1]), N)
        ind = range(np.shape(X)[1])
        for j in remove_idx:
            ind.remove(j)
        X_sample = X[:,ind]
        clf.fit(X_sample[subIndex,:],y[subIndex])
#        errorMatrix[i,0] = clf.score(X_sample[subIndex,:],DA_output[subIndex])
#        errorMatrix[i,1] = clf.score(X_sample[range(150,250),:],DA_output[range(150,250)])
        testIndex1 = range(30,45)+range(80,95)
        testIndex2 = range(180,195)+range(230,145)
        errorMatrix[i,0] = clf.score(X_sample[testIndex1,:],y[testIndex1])
        errorMatrix[i,1] = clf.score(X_sample[testIndex2,:],y[testIndex2])
      
        #errorMatrix[i,2] = clf.score(X_sample[range(100,150),:],DA_output[range(100,150)])
    return errorMatrix
