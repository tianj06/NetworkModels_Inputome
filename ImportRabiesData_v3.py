# -*- coding: utf-8 -*-
"""
Import matlab formatted data 
Ju Tian 2015.5
"""
import scipy.io as sio
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from RabiesDataLinearModel import *
#%% load data
file_name = r'C:\Users\uchidalab\Dropbox (Uchida Lab)\lab\FunInputome\NetworkModels\allUnitrawPSTH'
d = sio.loadmat(file_name)
psthInputs = d["rawpsthAll"]
brainArea = d["brainArea"]
brainArea = brainArea.flatten()
temp = []
for i in xrange(len(brainArea)):
    temp.append(brainArea[i][0])
brainArea = np.array(temp)
#%% extract X and y for later fitting
psthDA = np.nanmean(psthInputs[brainArea == "Dopamine",:], axis=0)
remoteUnits = (brainArea != "Dopamine") & (brainArea != "VTA type2") & (brainArea != "VTA type3")
psthInputs_remote = psthInputs[remoteUnits,:]
brainAreaInputs = brainArea[remoteUnits]
# sort input matrix by brain area
sortIdx = np.argsort(np.array(brainAreaInputs))
brainAreaInputs = brainAreaInputs[sortIdx]
psthInputs_remote = psthInputs_remote[sortIdx,:] 
# factorize brain area
brainAreaCat, brainAreaCode = pd.factorize(brainAreaInputs)

#%% preprocessing step 1: get subset of data by trial Type
trialName = {'90%water':0, '50% reward':1, '10% reward':2, '80% airpuff':3,	
             'omission 90% water':4, 'omission 50% reward':5,	'omssion 10% reward':6,
             'omission airpuff':7,	'free reward':8,	'free airpuff':9}
removeType = [2,7,8,9]
Idx = set(trialName.values()) - set(removeType)
RemainIndex = list()
for i in Idx:
    RemainIndex.extend(range(i*50,(i+1)*50)) 
All_input = preprocessData(psthInputs_remote.T,RemainIndex)
psthDA = psthDA.reshape(-1,1);
DA_output = preprocessData(psthDA,RemainIndex)
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
Ninputs = len(brainAreaInputs);
for i in range(1,Ninputs,5):
    (train_error,min_error_index)= bootstrp_fitting_error(All_input,DA_output,clf, n=i,N=500)         
    meanError.append(np.mean(train_error))
    stdError.append(np.std(train_error))
    bestSubGroup.append(min_error_index)
    minError.append(min(train_error))
    
plt.figure()    
plt.errorbar(range(1,Ninputs,5), meanError, stdError)
plt.plot(range(1,Ninputs,5),minError)
plt.legend(['averag fit error','min fit error'])
plt.xlabel('Number of neurons')
plt.ylabel('Mean squared error')

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

#%% leave N out to look at the stability of weights
# plot weight profile
meane = list()
stde = list()
for i in [5,20,50,90,130]:
    a = weightStability(All_input,DA_output, clf, N=i)
    meane.append(np.mean(a))
    stde.append(np.std(a))
plt.figure()
plt.bar(range(5), meane,yerr=stde)
plt.xticks(range(5),('5', '20', '50','90','130'))
plt.ylabel('correlation coefficients of weights')
plt.xlabel('number of neurons left out')

#%% since the weights looks pretty stable, now look at coefficients distribution
weigthDistributionPlot(All_input[:,bestSubGroup[12]],DA_output,
                       GroupFactor=brainAreaCat[bestSubGroup[12]])
savePath = r'C:\Users\uchidalab\Dropbox (Uchida Lab)\lab\FunInputome\writing\Figures'

coef = weigthDistributionPlot(All_input,DA_output,clf,brainAreaInputs)
plt.savefig(savePath+"fig_final.svg")

plt.figure()
plt.hist(coef)
plt.xlabel('Weight')
plt.ylabel('Counts of neurons')

#%% 
trialtypeNames = ['90% W','50% W','80% Puff','OM 90% W','OM 50% W','OM 10% W'] 
# by brain area    
f, axes=plt.subplots(nrows=len(brainAreaCode),sharey=True)    
for i in range(len(brainAreaCode)):
    subsetPredictionPlot(All_input,DA_output,coef,
                         np.array(brainAreaInputs) == brainAreaCode[i],axes[i]) 
    axes[i].set_title( brainAreaCode[i])
    axes[i].set_xticklabels([])
    #axes[i].set_ylim([0.0,0.4])
    if i==len(brainAreaCode)-1:
        for j in range(6):
            axes[i].text(12+j*50,-0.2,trialtypeNames[j])
plt.savefig(savePath+"\linear_model_byarea.svg")

# by weights
f, axes=plt.subplots(nrows=2,sharey=True)    
subsetPredictionPlot(All_input,DA_output,coef, coef>0.05,axes[0]) 
axes[0].set_xticklabels([])
plt.title('coef>0.05 n= {0}'.format(sum(coef>0.05)))

subsetPredictionPlot(All_input,DA_output,coef, coef<=0.05,axes[1]) 
plt.title('coef<=0.05 n= {}'.format(sum(coef<=0.05)))

axes[1].set_xticklabels([])
for j in range(6):
    axes[1].text(12+j*50,-0.5,trialtypeNames[j])
plt.show()
#%% plot the individual psth of all top weighted neurons
clf.fit(All_input,DA_output)
coef = clf.coef_
coef = coef.ravel()
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
plt.show()
#%% for each input area fit DA responses and plot the fitted result
f, axes=plt.subplots(nrows=len(brainAreaCode),sharey=True)    
e = []
for i in range(len(brainAreaCode)):
    neuronIdx = np.array(brainAreaInputs) == brainAreaCode[i]
    X = All_input[:,neuronIdx]
    y = DA_output
    coef_temp = clf.fit(X,y)    
    R2 = metrics.explained_variance_score(clf.predict(X),y) 
    e.append(R2)
    axes[i].plot(DA_output)
    axes[i].plot(clf.predict(X))
    axes[i].legend(['DA','predicted'],bbox_to_anchor=(1.25, 1.1))
    
    axes[i].set_title( '{0},R2={1}'.format(brainAreaCode[i],R2))
    axes[i].set_xticklabels([])
    if i==len(brainAreaCode)-1:
        for j in range(6):
            axes[i].text(12+j*50,-0.2,trialtypeNames[j])
plt.show()

plt.figure()
plt.bar(e)
#%% plot example psths to show the diversity of neuronal responses
plotTrialTypes = [range(50*6,50*7), range(50),range(50,50+50)]
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
  