
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
#reading csv files of train and test
dfTrain = pd.read_csv('/Users/ishaan/Desktop/256-Prog-Assig-2/train.csv')
dfTest = pd.read_csv('/Users/ishaan/Desktop/256-Prog-Assig-2/test.csv')
dfSubmission = pd.read_csv('/Users/ishaan/Desktop/256-Prog-Assig-2/sampleSubmission.csv')


# In[10]:



#ignoring all NA values of despicable me in train
dfTrain = dfTrain[np.isfinite(dfTrain['Rating [Despicable Me]'])]
originalDfTrain=dfTrain[np.isfinite(dfTrain['Rating [Despicable Me]'])]
#calculating mean of all rating columns(Train)
mean_StarWars=dfTrain['Rating [Rogue One/Star Wars]'].mean()
mean_FightClub=dfTrain['Rating [Fight Club]'].mean()
mean_TheLordoftheRings=dfTrain['Rating [The Lord of the Rings]'].mean()
mean_Trolls=dfTrain['Rating [Trolls]'].mean()
mean_DespicableMe=dfTrain['Rating [Despicable Me]'].mean()
mean_Kubo=dfTrain['Rating [Kubo]'].mean()
mean_TheHangover=dfTrain['Rating [The hangover]'].mean()
mean_CaptainAmerica=dfTrain['Rating [Captain America]'].mean()
mean_TheBigLebowski=dfTrain['Rating [The big Lebowski]'].mean()
mean_AlmostFamous=dfTrain['Rating [Almost famous]'].mean()
mean_TheHungerGames=dfTrain['Rating [The hunger games]'].mean()
mean_PulpFiction=dfTrain['Rating [Pulp Fiction]'].mean()
mean_500DaysOfSummer=dfTrain['Rating [500 days of Summer]'].mean()
mean_Twilight=dfTrain['Rating [Twilight]'].mean()
mean_LalaLand=dfTrain['Rating [Lala Land]'].mean()

#calculating mean of all rating columns(Test)
meanTest_StarWars=dfTest['Rating [Rogue One/Star Wars]'].mean()
meanTest_FightClub=dfTest['Rating [Fight Club]'].mean()
meanTest_TheLordoftheRings=dfTest['Rating [The Lord of the Rings]'].mean()
meanTest_Trolls=dfTest['Rating [Trolls]'].mean()
meanTest_Kubo=dfTest['Rating [Kubo]'].mean()
meanTest_TheHangover=dfTest['Rating [The hangover]'].mean()
meanTest_CaptainAmerica=dfTest['Rating [Captain America]'].mean()
meanTest_TheBigLebowski=dfTest['Rating [The big Lebowski]'].mean()
meanTest_AlmostFamous=dfTest['Rating [Almost famous]'].mean()
meanTest_TheHungerGames=dfTest['Rating [The hunger games]'].mean()
meanTest_PulpFiction=dfTest['Rating [Pulp Fiction]'].mean()
meanTest_500DaysOfSummer=dfTest['Rating [500 days of Summer]'].mean()
meanTest_Twilight=dfTest['Rating [Twilight]'].mean()
meanTest_LalaLand=dfTest['Rating [Lala Land]'].mean()

#subtract mean from every ratings column to normalize(Train)
dfTrain['Rating [Rogue One/Star Wars]']=dfTrain['Rating [Rogue One/Star Wars]']-mean_StarWars
dfTrain['Rating [Fight Club]']=dfTrain['Rating [Fight Club]']-mean_FightClub
dfTrain['Rating [The Lord of the Rings]']=dfTrain['Rating [The Lord of the Rings]']-mean_TheLordoftheRings
dfTrain['Rating [Trolls]']=dfTrain['Rating [Trolls]']-mean_Trolls
dfTrain['Rating [Despicable Me]']=dfTrain['Rating [Despicable Me]']-mean_DespicableMe
dfTrain['Rating [Kubo]']=dfTrain['Rating [Kubo]']-mean_Kubo
dfTrain['Rating [The hangover]']=dfTrain['Rating [The hangover]']-mean_TheHangover
dfTrain['Rating [Captain America]']=dfTrain['Rating [Captain America]']-mean_CaptainAmerica
dfTrain['Rating [The big Lebowski]']=dfTrain['Rating [The big Lebowski]']-mean_TheBigLebowski
dfTrain['Rating [Almost famous]']=dfTrain['Rating [Almost famous]']-mean_AlmostFamous
dfTrain['Rating [The hunger games]']=dfTrain['Rating [The hunger games]']-mean_TheHungerGames
dfTrain['Rating [Pulp Fiction]']=dfTrain['Rating [Pulp Fiction]']-mean_PulpFiction
dfTrain['Rating [500 days of Summer]']=dfTrain['Rating [500 days of Summer]']-mean_500DaysOfSummer
dfTrain['Rating [Twilight]']=dfTrain['Rating [Twilight]']-mean_Twilight
dfTrain['Rating [Lala Land]']=dfTrain['Rating [Lala Land]']-mean_LalaLand


#subtract mean from every ratings column to normalize(Test)
dfTest['Rating [Rogue One/Star Wars]']=dfTest['Rating [Rogue One/Star Wars]']-meanTest_StarWars
dfTest['Rating [Fight Club]']=dfTest['Rating [Fight Club]']-meanTest_FightClub
dfTest['Rating [The Lord of the Rings]']=dfTest['Rating [The Lord of the Rings]']-meanTest_TheLordoftheRings
dfTest['Rating [Trolls]']=dfTest['Rating [Trolls]']-meanTest_Trolls
dfTest['Rating [Kubo]']=dfTest['Rating [Kubo]']-meanTest_Kubo
dfTest['Rating [The hangover]']=dfTest['Rating [The hangover]']-meanTest_TheHangover
dfTest['Rating [Captain America]']=dfTest['Rating [Captain America]']-meanTest_CaptainAmerica
dfTest['Rating [The big Lebowski]']=dfTest['Rating [The big Lebowski]']-meanTest_TheBigLebowski
dfTest['Rating [Almost famous]']=dfTest['Rating [Almost famous]']-meanTest_AlmostFamous
dfTest['Rating [The hunger games]']=dfTest['Rating [The hunger games]']-meanTest_TheHungerGames
dfTest['Rating [Pulp Fiction]']=dfTest['Rating [Pulp Fiction]']-meanTest_PulpFiction
dfTest['Rating [500 days of Summer]']=dfTest['Rating [500 days of Summer]']-meanTest_500DaysOfSummer
dfTest['Rating [Twilight]']=dfTest['Rating [Twilight]']-meanTest_Twilight
dfTest['Rating [Lala Land]']=dfTest['Rating [Lala Land]']-meanTest_LalaLand


# In[11]:


#Filling all NaN values to 0 for cosine similarity
dfTrain=dfTrain.fillna(0)
dfTest=dfTest.fillna(0)


# In[19]:


#subset
subsetTrain=dfTrain.iloc[:,5:21]
subsetTest=dfTest.iloc[:,4:21]

def cos_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

trainList=[]
testList=[]

for row in subsetTrain.iterrows():
    index, data = row
    trainList.append(data.tolist())
    
for row in subsetTest.iterrows():
    index, data = row
    testList.append(data.tolist())

maxi=0
val=0
index=0
maxList=[]
maxIndex=[]
for i in range(0, len(testList)):
    for j in range(0, len(trainList)):
        val=cos_similarity(testList[i],trainList[j])
        maxi=max(maxi,val)
        if(val==maxi):
            index=j+1        
    maxList.append(maxi)
    maxIndex.append(index)
    index=0
    maxi=0
pd.options.display.max_rows = 4000
pred=[]
sin=originalDfTrain['Rating [Despicable Me]']
for i in range(0, len(maxIndex)):
    pred.append(sin.iat[maxIndex[i]-1,])
    
dfSubmission['Rating [Despicable Me]']=pred
print (dfSubmission)
dfSubmission.to_csv('/Users/ishaan/Desktop/256-Prog-Assig-2/sampleSubmission9.csv',index=False)

