
# coding: utf-8

# In[81]:


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# In[82]:


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else: 
            print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# In[83]:


import numpy as np
def train(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #change to log()
    p0Vect = np.log(p0Num/p0Denom) #change to log()
    return p0Vect,p1Vect,pAbusive


# In[88]:


def test(sentences,classes,testEntry1,testEntry2):
    listOPosts,listClasses = sentences,classes
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = train(np.array(trainMat),np.array(listClasses))
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry1))
    print (testEntry1,'classified as: ',classify(thisDoc,p0V,p1V,pAb))
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry2))
    print(testEntry2,"classified as:",classify(thisDoc,p0V,p1V,pAb))


# In[85]:


def classify(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 'Positive'
    else:
        return 'Negative'

