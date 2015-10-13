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

#%% load data
file_name = r'C:\Users\Hideyuki\Dropbox (Uchida Lab)\lab\FunInputome\rabies\allIdentified\allUnitrawPSTH'
d = sio.loadmat(file_name)
psthInputs = d["rawpsthAll"]
brainArea = d["brainArea"]

#%% extract X and y for later fitting
psthDA = np.nanmean(psthInputs[np.squeeze(brainArea == "DA"),:], axis=0)
remoteUnits = (brainArea != "DA") & (brainArea != "VTA2") & ((brainArea != "VTA3"))
psthInputs_remote = psthInputs[np.squeeze(remoteUnits),:]
# sort input matrix by brain area
brainAreaInputs = list()
for i in range(0,124):
    brainAreaInputs.append(brainArea[0,i][0])
temp = pd.factorize(brainAreaInputs)
brainAreaCode = temp[1]
brainAreaCat = temp[0]

psthInputs_remote = psthInputs_remote[np.argsort(np.array(brainAreaCat)),:] 

#%% preprocessing step 1: get subset of data by trial Type
trialName = {'90%water':0, '50% reward':1, '10% reward':2, '80% airpuff':3,	
             'omission 90% water':4, 'omission 50% reward':5,	'omssion 10% reward':6,
             'omission airpuff':7,	'free reward':8,	'free airpuff':9}

Idx = set(trialName.values()) - set([2,7,8,9])
a = list()
for i in Idx:
    a.extend(range(i*50,(i+1)*50))

psthInputs_remote_subTypes= np.array(psthInputs_remote)[:,a]

#%% preprocessing step 2: fill in missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
psthInputs_remote_subTypes = imp.fit_transform(psthInputs_remote_subTypes.T)

#%% preprocessing step 3: fill scale data
min_max_scaler = preprocessing.MinMaxScaler()
psthInputs_remote_subTypes_processed =min_max_scaler.fit_transform(psthInputs_remote_subTypes)
psthDA_processed = min_max_scaler.fit_transform(psthDA[a])

#plt.plot(psthInputs_remote_subTypes_processed[:,::10])
#plt.plot(psthDA_processed)
#plt.show()

#%% setup a linear model
clf = linear_model.Ridge(alpha=0.5) #LinearRegression() #

#%% bootstrap the fitting error
meanError = list()
stdError = list()
bestSubGroup = list()
minError = list()
for i in range(1,124,5):
    train_error = list()
    min_train_error = 10000
    for j in range(0,500):
        ind = random.sample(range(1, 124), i)
        X_sample = psthInputs_remote_subTypes_processed[:,ind]
        clf.fit(X_sample,psthDA_processed)
        e = metrics.mean_squared_error(clf.predict(X_sample),psthDA_processed)    
        train_error.append(e)
        if e<min_train_error:
            min_train_error = e
            min_error_index = ind 
             
    meanError.append(np.mean(train_error))
    stdError.append(np.std(train_error))
    bestSubGroup.append(min_error_index)
    minError.append(min_train_error)
    
plt.figure()    
plt.errorbar(range(1,124,5), meanError, stdError)
plt.plot(range(1,124,5),minError)
plt.legend(['averag fit error','min fit error'])
plt.xlabel('Number of neurons')
plt.ylabel('Mean squared error')

#%% plot the most frequent units for good fitting
neuronCount = collections.Counter([item for sublist in bestSubGroup for item in sublist])
np.sort(neuronCount.values())
f,axes = plt.subplots(4,3)
f.subplots_adjust(hspace=.5)
NeuronId = neuronCount.keys()
NeuronCounts = neuronCount.values()
plotIndex = np.array(NeuronId)[np.array(NeuronCounts)>16]
c = np.array(NeuronCounts)[np.array(NeuronCounts)>16]
plotData = psthInputs_remote[plotIndex,:]
plotBrainArea = np.array(brainAreaInputs)[plotIndex]
plotcount = 0
for i in range(4):
    for j in range(3):
        if plotcount<len(plotIndex):
            l1 = axes[i,j].plot(plotData[plotcount,range(50*6,50*7)])
            l2 = axes[i,j].plot(plotData[plotcount,range(50)])
            l3 = axes[i,j].plot(plotData[plotcount,range(50,50+50)])
            axes[i,j].set_title('%s freq:%.2f' %(plotBrainArea[plotcount], float(c[plotcount])/len(bestSubGroup)))
            axes[i,j].set_xticklabels(['','0','1','2','3',''])
            plotcount+=1
plt.figlegend( (l2, l3, l1), ('90% Reward', '50% Reward', 'No Reward'), 'upper right' )
plt.show()


#%% plot the fitted psth given a specific number of inputs. (not useful)
def bootpsth(N=10):
    X_sample = psthInputs_remote_subTypes_processed[:,random.sample(range(1, 124), N)]
    clf.fit(X_sample,psthDA_processed)
    predicted_PSTH = clf.predict(X_sample)
    plt.plot(psthDA_processed)
    plt.plot(predicted_PSTH)
    plt.ylabel('normalized response')
    plt.legend(['DA','predicted'])
    
bootpsth(N=10)

#%% leave N out to look at the stability of weights

def weightStabilityPlot(N=5):
    coefMatrix=np.full([124,100],np.nan,dtype='float')
    for i in range(100):
        remove_idx = random.sample(range(1, 124), N)
        ind = range(124)
        for j in remove_idx:
            ind.remove(j)
        X_sample = psthInputs_remote_subTypes_processed[:,ind]
        clf.fit(X_sample,psthDA_processed)
        coefMatrix[ind,i]=clf.coef_
    meanCoef=np.nanmean(coefMatrix,axis=1)
    normalizedError = list()
    for i in range(100):
        NanIdx = np.isnan(coefMatrix[:,i])
        #ne = np.linalg.norm( coefMatrix[~NanIdx,i] - meanCoef[~NanIdx])/np.linalg.norm(meanCoef[~NanIdx])
        ne = np.corrcoef(coefMatrix[~NanIdx,i],meanCoef[~NanIdx])
        normalizedError.append(ne)
    return normalizedError
#    plt.pcolor(coefMatrix,cmap='RdBu',vmin=-0.1, vmax=0.1)
#    plt.colorbar()
#    plt.xlabel('Subgroup ID')
#    plt.ylabel('Neuron ID')
#    plt.title('%d Neurons are omitted' %N)
#    plt.xlim((0,100))
#    plt.ylim((0,124))
#    plt.show()

# plot weight profile
meane = list()
stde = list()
for i in [5,20,50,70,90]:
    a = weightStabilityPlot(N=i)
    meane.append(np.mean(a))
    stde.append(np.std(a))
plt.figure()
plt.bar(range(5), meane,yerr=stde)
plt.xticks(range(5),('5', '20', '50','70','90'))
plt.ylabel('correlation coefficients of weights')
plt.xlabel('number of neurons left out')

#%% since the weights looks pretty stable, now look at coefficients distribution
def weigthDistributionPlot(neuronIdx):
    X_sample = psthInputs_remote_subTypes_processed[:,neuronIdx]
    clf.fit(X_sample,psthDA_processed)
    # plot fitting
    f,ax = plt.subplots(2)
    ax[0].plot(clf.predict(X_sample))
    ax[0].plot(psthDA_processed)
    ax[0].legend(['DA','predicted'])
    ax[0].set_ylabel('Normalized firing rate')
    ax[0].set_title('N inputs = %d' %len(neuronIdx))
    
    coef = clf.coef_   
    sns.violinplot(pd.Series(coef),groupby=brainAreaCat[neuronIdx],inner='points',
                   names=brainAreaCode,ax=ax[1])
    ax[1].set_ylabel('Coefficients')
    return(coef)

weigthDistributionPlot(bestSubGroup[7])
    
weigthDistributionPlot(bestSubGroup[10])

coef = weigthDistributionPlot(range(len(brainAreaCat)))

sortIdx = np.argsort(coef)
coef[sortIdx]
plt.figure()
plt.hist(coef)

def subsetPredictionPlot(neuronIdx):
    trialtypeNames = ['90% W','50% W','80% Puff','OM 90% W','OM 50% W','OM 10% W']
    subPredict = np.average(psthInputs_remote_subTypes_processed[:,neuronIdx], axis=1,weights=coef[neuronIdx])
    plt.figure()
    plt.plot(psthDA_processed)
    plt.plot(subPredict)
    plt.legend(['DA','predicted'])
    for i in range(6):
        plt.text(25+i*50,0.8,trialtypeNames[i])
    
for i in range(len(brainAreaCode)):
    subsetPredictionPlot(np.array(brainAreaInputs) == brainAreaCode[i]) 
    plt.title( brainAreaCode[i])
    


# plot the individual psth of all top weighted neurons

np.sum(coef>0.05)

f,axes = plt.subplots(4,3)
f.subplots_adjust(hspace=.5)
plotData = psthInputs_remote[coef>0.05,:]
plotBrainArea = np.array(brainAreaInputs)[coef>0.05]
plotcount = 0
for i in range(4):
    for j in range(3):
        if plotcount<len(plotBrainArea):
            l1 = axes[i,j].plot(plotData[plotcount,range(50*6,50*7)])
            l2 = axes[i,j].plot(plotData[plotcount,range(50)])
            l3 = axes[i,j].plot(plotData[plotcount,range(50,50+50)])
            axes[i,j].set_title('%s weight %.3f' %(plotBrainArea[plotcount] , coef[coef>0.05][plotcount]))
            axes[i,j].set_xticklabels(['','0','1','2','3',''])
            plotcount+=1
plt.figlegend( (l2, l3, l1), ('90% Reward', '50% Reward', 'No Reward'), 'upper right' )
plt.show()

# use 90% and 50% and 0% data to predict airpuff and omission data
subIndex = range(100)+range(50*5,50*6)
X_sample = psthInputs_remote_subTypes_processed[subIndex,:]
clf.fit(X_sample,psthDA_processed[subIndex])
f,ax = plt.subplots(2)
ax[0].plot(psthDA_processed[subIndex])
ax[0].plot(clf.predict(psthInputs_remote_subTypes_processed[subIndex,:]))
ax[0].legend(['DA','predicted'])
ax[0].set_ylabel('Normalized firing rate')
ax[1].plot(psthDA_processed[range(100,250)])
ax[1].plot(clf.predict(psthInputs_remote_subTypes_processed[range(100,250),:]))
ax[1].legend(['DA','predicted'])
ax[1].set_ylabel('Normalized firing rate')


plt.figure()
plt.plot(psthDA_processed)


def crossPredictionStability(N=5):
    subIndex = range(100)+range(50*5,50*6)
    errorMatrix=np.zeros((100,2),dtype='float')
    for i in range(100):
        remove_idx = random.sample(range(1, 124), N)
        ind = range(124)
        for j in remove_idx:
            ind.remove(j)
        X_sample = psthInputs_remote_subTypes_processed[:,ind]
        clf.fit(X_sample[subIndex,:],psthDA_processed[subIndex])
        errorMatrix[i,0] = clf.score(X_sample[subIndex,:],psthDA_processed[subIndex])
        errorMatrix[i,1] = clf.score(X_sample[range(150,250),:],psthDA_processed[range(150,250)])
        #errorMatrix[i,2] = clf.score(X_sample[range(100,150),:],psthDA_processed[range(100,150)])

    return errorMatrix

meane = np.zeros((5,2))
stde = np.zeros((5,2))
n=0
for i in [5,20,50,70,90]:
    e_R2 = crossPredictionStability(N=i)
    meane[n,:] = np.mean(e_R2,axis=0)
    stde[n,:] = np.std(e_R2,axis=0)
    n = n+1    
plt.figure()
plt.errorbar(range(5), meane[:,0],stde[:,0])
plt.errorbar(range(5), meane[:,1],stde[:,1])

plt.xticks(range(5),('5', '20', '50','70','90'))
plt.ylabel('R2')
plt.xlabel('number of neurons left out')
plt.legend(('Reward (fitted)','Omission (new)'))
