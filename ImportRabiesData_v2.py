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

#%% load data
file_name = r'D:\Dropbox (Uchida Lab)\lab\FunInputome\rabies\allIdentified\allUnitrawPSTH'
d = sio.loadmat(file_name)
psthInputs = d["rawpsthAll"]
brainArea = d["brainArea"]

#%% extract X and y for later fitting
psthDA = np.nanmean(psthInputs[np.squeeze(brainArea == "DA"),:], axis=0)
remoteUnits = (brainArea != "DA") & (brainArea != "VTA2") & (brainArea != "VTA3")
psthInputs_remote = psthInputs[np.squeeze(remoteUnits),:]
# sort input matrix by brain area
brainAreaInputs = list()
for i in range(124):
    brainAreaInputs.append(brainArea[0,i][0])
temp = pd.factorize(brainAreaInputs)
brainAreaCode = temp[1]
brainAreaCat = temp[0]
sortIdx = np.argsort(np.array(brainAreaCat))
brainAreaInputs = np.array(brainAreaInputs)[sortIdx] 
brainAreaCat = brainAreaCat[sortIdx]
psthInputs_remote = psthInputs_remote[sortIdx,:] 
#%% preprocessing step 1: get subset of data by trial Type
trialName = {'90%water':0, '50% reward':1, '10% reward':2, '80% airpuff':3,	
             'omission 90% water':4, 'omission 50% reward':5,	'omssion 10% reward':6,
             'omission airpuff':7,	'free reward':8,	'free airpuff':9}

def preprocessData(removeType = [2,7,8,9]):
    Idx = set(trialName.values()) - set(removeType)
    a = list()
    for i in Idx:
        a.extend(range(i*50,(i+1)*50))
    
    psthInputs_remote_subTypes= np.array(psthInputs_remote)[:,a]
    
    # preprocessing step 2: fill in missing values
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    psthInputs_remote_subTypes = imp.fit_transform(psthInputs_remote_subTypes.T)
    
    # preprocessing step 3: fill scale data
    min_max_scaler = preprocessing.MinMaxScaler()
    psthInputs_remote_subTypes_processed =min_max_scaler.fit_transform(psthInputs_remote_subTypes)
    psthDA_processed = min_max_scaler.fit_transform(psthDA[a])
    return (psthInputs_remote_subTypes_processed,psthDA_processed)

(All_input,DA_output) = preprocessData()
#plt.plot(psthInputs_remote_subTypes_processed[:,::10])
#plt.plot(psthDA_processed)
#plt.show()

#%% setup a linear model
clf = linear_model.Ridge(alpha=0.5) #LinearRegression() #

#%% bootstrap the fitting error
def bootstrp_fitting_error(X,y,n=50,N=500):
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
    
meanError = list()
stdError = list()
bestSubGroup = list()
minError = list()

for i in range(1,124,5):
    (train_error,min_error_index)= bootstrp_fitting_error(X=All_input,y=DA_output,n=i,N=500)         
    meanError.append(np.mean(train_error))
    stdError.append(np.std(train_error))
    bestSubGroup.append(min_error_index)
    minError.append(min(train_error))
    
plt.figure()    
plt.errorbar(range(1,124,5), meanError, stdError)
plt.plot(range(1,124,5),minError)
plt.legend(['averag fit error','min fit error'])
plt.xlabel('Number of neurons')
plt.ylabel('Mean squared error')

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
#%%
freqCount = countTopFrequentUnits(bestSubGroup)

selectedPSTH = psthInputs_remote[freqCount[:,0].astype(int),:]
plotTrialTypes = [range(50*6,50*7), range(50),range(50,50+50)]
plotTrialNames = ['90% Reward', '50% Reward', 'No Reward']
plotList = list()
titleList = list()
xtickLabelList = list()
plotBrainArea = np.array(brainAreaInputs)[freqCount[:,0].astype(int)]
for i in range(np.shape(selectedPSTH)[0]):    
    plotList.append([selectedPSTH[i,l] for l in plotTrialTypes]) 
    titleList.append( '%s freq:%.2f' %(plotBrainArea[i],freqCount[i,1]))
    xtickLabelList.append(['','0','1','2','3',''])
(plotF,plotAxes)= panelPSTHplotting(plotList,subtitles=titleList,xtickLabels=xtickLabelList)



#%% plot the fitted psth given a specific number of inputs. (not useful)
def bootpsth(N=10):
    X_sample = All_input[:,random.sample(range(1, 124), N)]
    clf.fit(X_sample,DA_output)
    predicted_PSTH = clf.predict(X_sample)
    plt.plot(DA_output)
    plt.plot(predicted_PSTH)
    plt.ylabel('normalized response')
    plt.legend(['DA','predicted'])
    
bootpsth(N=10)

#%% leave N out to look at the stability of weights

def weightStability(X,y,N=5):
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
    a = weightStability(All_input,DA_output, N=i)
    meane.append(np.mean(a))
    stde.append(np.std(a))
plt.figure()
plt.bar(range(5), meane,yerr=stde)
plt.xticks(range(5),('5', '20', '50','70','90'))
plt.ylabel('correlation coefficients of weights')
plt.xlabel('number of neurons left out')

#%% since the weights looks pretty stable, now look at coefficients distribution
def weigthDistributionPlot(X,y,GroupFactor):
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

#weigthDistributionPlot(All_input[:,bestSubGroup[7]])    
#weigthDistributionPlot(All_input[:,bestSubGroup[10]])


weigthDistributionPlot(All_input[:,bestSubGroup[12]],DA_output,
                       GroupFactor=brainAreaCat[bestSubGroup[12]])

coef = weigthDistributionPlot(All_input,DA_output,GroupFactor=brainAreaCat)


plt.figure()
plt.hist(coef)
plt.xlabel('Weight')
plt.ylabel('Counts of neurons')
#%% 
trialtypeNames = ['90% W','50% W','80% Puff','OM 90% W','OM 50% W','OM 10% W']
def subsetPredictionPlot(neuronIdx,ax,X=All_input,Weights=coef):
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
# by brain area    
f, axes=plt.subplots(nrows=len(brainAreaCode),sharey=True)    
for i in range(len(brainAreaCode)):
    subsetPredictionPlot(np.array(brainAreaInputs) == brainAreaCode[i],axes[i]) 
    axes[i].set_title( brainAreaCode[i])
    axes[i].set_xticklabels([])
    if i==len(brainAreaCode)-1:
        for j in range(6):
            axes[i].text(12+j*50,-0.2,trialtypeNames[j])
plt.show()

# by weights
f, axes=plt.subplots(nrows=2,sharey=True)    
subsetPredictionPlot(coef>0.05,axes[0]) 
axes[0].set_xticklabels([])
subsetPredictionPlot(coef<=0.05,axes[1]) 
axes[1].set_xticklabels([])
for j in range(6):
    axes[1].text(12+j*50,-0.5,trialtypeNames[j])
plt.show()
#%% plot the individual psth of all top weighted neurons
clf.fit(All_input,DA_output)
coef = clf.coef_
plotTrialTypes = [range(50*6,50*7), range(50),range(50,50+50)]
plotTrialNames = ['90% Reward', '50% Reward', 'No Reward']
plotList = list()
titleList = list()
xtickLabelList = list()
plotBrainArea = np.array(brainAreaInputs)[coef>0.05]
selectedPSTH = psthInputs_remote[coef>0.05,:]
for i in range(np.shape(selectedPSTH)[0]):    
    plotList.append([selectedPSTH[i,l] for l in plotTrialTypes]) 
    titleList.append( '%s w= %.2f' %(plotBrainArea[i],coef[coef>0.05][i]))
    xtickLabelList.append(['','0','1','2','3',''])
            
panelPSTHplotting(plotList,subtitles=titleList,xtickLabels=xtickLabelList)


tempIdx = [3,4,6,8,9,10,12,14,27,31,33,66]
a = sortIdx.astype(int).tolist()
subIdx = [a.index(i) for i in tempIdx]
plotList = list()
selectedPSTH = psthInputs_remote[subIdx,:]
for i in range(np.shape(selectedPSTH)[0]):    
    plotList.append([selectedPSTH[i,l] for l in plotTrialTypes]) 
panelPSTHplotting(plotList,xtickLabels=xtickLabelList)



#%% use 90% and 50% and 0% data to predict airpuff and omission data
subIndex = range(100)+range(50*5,50*6)
X_sample = All_input[subIndex,:]
clf.fit(X_sample,DA_output[subIndex])
f,ax = plt.subplots(2)
ax[0].plot(DA_output[subIndex])
ax[0].plot(clf.predict(All_input[subIndex,:]))
ax[0].legend(['DA','predicted'])
ax[0].set_ylabel('Normalized firing rate')
ax[1].plot(DA_output[range(100,250)])
ax[1].plot(clf.predict(All_input[range(100,250),:]))
ax[1].legend(['DA','predicted'])
ax[1].set_ylabel('Normalized firing rate')


plt.figure()
plt.plot(DA_output)

#%%
def crossPredictionStability(N=5,X=All_input):
    subIndex = range(100)+range(50*5,50*6)
    errorMatrix=np.zeros((100,2),dtype='float')
    for i in range(100):
        remove_idx = random.sample(range(1, np.shape(X)[1]), N)
        ind = range(np.shape(X)[1])
        for j in remove_idx:
            ind.remove(j)
        X_sample = X[:,ind]
        clf.fit(X_sample[subIndex,:],DA_output[subIndex])
#        errorMatrix[i,0] = clf.score(X_sample[subIndex,:],DA_output[subIndex])
#        errorMatrix[i,1] = clf.score(X_sample[range(150,250),:],DA_output[range(150,250)])
        testIndex1 = range(30,45)+range(80,95)
        testIndex2 = range(180,195)+range(230,145)
        errorMatrix[i,0] = clf.score(X_sample[testIndex1,:],DA_output[testIndex1])
        errorMatrix[i,1] = clf.score(X_sample[testIndex2,:],DA_output[testIndex2])
      
        #errorMatrix[i,2] = clf.score(X_sample[range(100,150),:],DA_output[range(100,150)])
    return errorMatrix

meane = np.zeros((5,2))
stde = np.zeros((5,2))
n=0
for i in [1,5,10,20,40]:
    e_R2 = crossPredictionStability(N=i)
    meane[n,:] = np.mean(e_R2,axis=0)
    stde[n,:] = np.std(e_R2,axis=0)
    n = n+1    
plt.figure(figsize=(4,3))
plt.tight_layout()
plt.errorbar(range(5), meane[:,0],stde[:,0])
plt.errorbar(range(5), meane[:,1],stde[:,1])
plt.xticks(range(5),('1', '5', '10','20','40'))
plt.ylabel('R2')
plt.xlabel('number of neurons left out')
plt.legend(('Reward (fitted)','Omission (new)'))
plt.xlim((-0.5,4.5))
#%% plot some examples
subIndex = range(100)+range(50*5,50*6)
clf.fit(All_input[subIndex,:],DA_output[subIndex])
f,axes = plt.subplots(2)
f.subplots_adjust(hspace=.6)

axes[0].plot(DA_output[subIndex])
axes[0].plot(clf.predict(All_input[subIndex,:]))
axes[0].set_xticks(range(0,len(subIndex),50))
axes[0].set_xticklabels([])
axes[0].xaxis.grid(linewidth=2)
axes[0].yaxis.grid(None)
axes[0].legend(('DA','fitted'))
axes[0].yaxis.set_major_locator(MaxNLocator(4))


newIndex = range(150,250)+range(100,150)
axes[1].plot(DA_output[newIndex])
axes[1].plot(clf.predict(All_input[newIndex,:]))
axes[1].set_xticks(range(0,len(newIndex),50))
axes[1].set_xticklabels([])
axes[1].xaxis.grid(linewidth=2)
axes[1].yaxis.grid(None)
axes[1].legend(('DA','fitted'))
axes[1].yaxis.set_major_locator(MaxNLocator(4))
axes[1].set_ylim((0,1))
#%% using not top rated 
RPEindexFile = r'C:\Users\Hideyuki\Dropbox (Uchida Lab)\lab\FunInputome\rabies\allIdentified\RPEind.txt'
RPEindex = np.loadtxt(RPEindexFile, dtype= int )
#subIdx = np.squeeze([np.where(sortIdx==i) for i in RPEindex])
#a = sortIdx.astype(int).tolist()
subIdx = RPEindex #[a.index(i) for i in RPEindex]
plotList = list()
titleList = list()
xtickLabelList = list()
selectedPSTH = psthInputs_remote[subIdx,:]
for i in range(len(subIdx)):    
    plotList.append([selectedPSTH[i,l] for l in plotTrialTypes]) 
    xtickLabelList.append(['','0','1','2','3',''])
            
panelPSTHplotting(plotList,xtickLabels=xtickLabelList) 






subInput = All_input[:,subIdx]
subGroup = brainAreaInputs[subIdx]
sub_coef = weigthDistributionPlot(subInput,DA_output,
                       GroupFactor=subGroup)
# error goes down as number of inputs increase                       
meanError = list()
stdError = list()
bestSubGroup = list()
minError = list()

for i in range(1,np.shape(subInput)[1],5):
    (train_error,min_error_index)= bootstrp_fitting_error(X=subInput,y=DA_output,n=i,N=500)         
    meanError.append(np.mean(train_error))
    stdError.append(np.std(train_error))
    bestSubGroup.append(min_error_index)
    minError.append(min(train_error))
    
plt.figure(figsize=(4,3))  
plt.tight_layout() 
plt.errorbar(range(1,np.shape(subInput)[1],5), meanError, stdError)
plt.plot(range(1,np.shape(subInput)[1],5),minError)
plt.legend(['averag fit error','min fit error'])
plt.xlabel('Number of neurons')
plt.ylabel('Mean squared error')                       

# stability plot                       
meane = list()
stde = list()
for i in [5,20,50,70,90]:
    a = weightStability(subInput,DA_output, N=i)
    meane.append(np.mean(a))
    stde.append(np.std(a))
plt.figure(figsize=(4,3))  
plt.tight_layout() 
plt.bar(range(5), meane,yerr=stde)
plt.xticks(range(5),('5', '20', '50','70','90'))
plt.ylabel('correlation coefficients of weights')
plt.xlabel('number of neurons left out')     

#
plt.figure(figsize=(4,3))
plt.hist(sub_coef)
plt.xlabel('Weight')
plt.ylabel('Counts of neurons')
plt.tight_layout()


# top weight neuron plot                       
plotList = list()
titleList = list()
xtickLabelList = list()
plotBrainArea = np.array(subGroup)[np.argsort(sub_coef)[:12]]
selectedPSTH = psthInputs_remote[coef<0.05,:][np.argsort(sub_coef)[:12],:]
for i in range(12):    
    plotList.append([selectedPSTH[i,l] for l in plotTrialTypes]) 
    titleList.append( '%s w= %.2f' %(plotBrainArea[i],sub_coef[i]))
    xtickLabelList.append(['','0','1','2','3',''])
            
panelPSTHplotting(plotList,subtitles=titleList,xtickLabels=xtickLabelList) 

# preiction by area
f, axes=plt.subplots(nrows=len(brainAreaCode),sharey=True)    
for i in range(len(brainAreaCode)):
    subsetPredictionPlot(np.array(subGroup) == brainAreaCode[i],axes[i],X=subInput,Weights=sub_coef) 
    axes[i].set_title( brainAreaCode[i])
    axes[i].set_xticklabels([])
    if i==len(brainAreaCode)-1:
        for j in range(6):
            axes[i].text(12+j*50,-0.2,trialtypeNames[j])
plt.show()     

# by weights
index_cag_list= [np.abs(sub_coef)>0.03,np.abs(sub_coef)<0.03] # ,sub_coef<-0.04
f, axes=plt.subplots(nrows=len(index_cag_list),sharey=True)  
plot_c = 0
for index in index_cag_list:
    subsetPredictionPlot(index,axes[plot_c],X=subInput,Weights=sub_coef) 
    axes[plot_c].set_xticklabels([])
    axes[plot_c].set_title('number of neuron: %d' %sum(index))
    plot_c+=1


# cross prediction
meane = np.zeros((5,2))
stde = np.zeros((5,2))
n=0
for i in [1,5,10,20,40]:
    e_R2 = crossPredictionStability(N=i,X=subInput)
    meane[n,:] = np.mean(e_R2,axis=0)
    stde[n,:] = np.std(e_R2,axis=0)
    n = n+1    
plt.figure(figsize=(4,3))
plt.errorbar(range(5), meane[:,0],stde[:,0])
plt.errorbar(range(5), meane[:,1],stde[:,1])
plt.xlim((-0.5,4.5)) 
plt.xticks(range(5),('1', '5', '10','20','40'))
plt.ylabel('R2')
plt.xlabel('number of neurons left out')
plt.legend(('Reward (fitted)','Omission (new)'))
plt.tight_layout()
  