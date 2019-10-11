# -*- coding: utf-8 -*-
"""
File:   hw02.py
Author: 
Date:   
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import collections


#plt.close('all') #close any open plots

""" =======================  Import DataSet ========================== """


Train_2D = np.loadtxt('2dDataSetforTrain.txt')
Train_7D = np.loadtxt('7dDataSetforTrain.txt')
Train_HS = np.loadtxt('HyperSpectralDataSetforTrain.txt')

labels_2D = Train_2D[:,Train_2D.shape[1]-1]
labels_7D = Train_7D[:,Train_7D.shape[1]-1]
labels_HS = Train_HS[:,Train_HS.shape[1]-1]

Train_2D = np.delete(Train_2D,Train_2D.shape[1]-1,axis = 1)
Train_7D = np.delete(Train_7D,Train_7D.shape[1]-1,axis = 1)
Train_HS = np.delete(Train_HS,Train_HS.shape[1]-1,axis = 1)

Test_2D = np.loadtxt('2dDataSetforTest.txt')
Test_7D = np.loadtxt('7dDataSetforTest.txt')
Test_HS = np.loadtxt('HyperSpectralDataSetforTest.txt')


""" ======================  Function definitions ========================== """

"""
===============================================================================
===============================================================================
======================== Probabilistic Generative Classfier ===================
===============================================================================
===============================================================================
"""

      

""" Here you can write functions to estimate the parameters for the training data, 
    and the prosterior probabilistic for the testing data. """
#Here I define 3 function to train data for PG, they are almost same but have some differences such as M.
def trainHSdataPGfull(Train_HS,labels_HS,diagonal=False):
    #define a function to train HS data
    Train = Train_HS#get Train data
    labels = labels_HS#get Train laabel
    Classes = np.sort(np.unique(labels))#get class
    M = 15
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
    X_train_class = []#initialize classification
    for j in range(Classes.shape[0]):
        #classify train data according to label
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]#get class
    class1=X_train_class[1]#get class
    class2=X_train_class[2]#get class
    class3=X_train_class[3]#get class
    class4=X_train_class[4]#get class
    mu0=np.mean(class0,axis=0)#get mean
    mu1=np.mean(class1,axis=0)#get mean
    mu2=np.mean(class2,axis=0)#get mean
    mu3=np.mean(class3,axis=0)#get mean
    mu4=np.mean(class4,axis=0)#get mean
    if diagonal==1:
        cov0 = np.cov(class0.T)*np.eye(class0.shape[1])#get diagonal matrix
        cov1 = np.cov(class1.T)*np.eye(class1.shape[1])#get diagonal matrix
        cov2 = np.cov(class2.T)*np.eye(class2.shape[1])#get diagonal matrix
        cov3 = np.cov(class3.T)*np.eye(class3.shape[1])#get diagonal matrix
        cov4 = np.cov(class4.T)*np.eye(class4.shape[1])#get diagonal matrix
    else:
        constant=1e-1
        cov0=np.cov(class0.T)+np.eye(class0.shape[1])*constant#get covariance and solve problem of singular matrix
        cov1=np.cov(class1.T)+np.eye(class1.shape[1])*constant#get covariance and solve problem of singular matrix
        cov2=np.cov(class2.T)+np.eye(class2.shape[1])*constant#get covariance and solve problem of singular matrix
        cov3=np.cov(class3.T)+np.eye(class3.shape[1])*constant#get covariance and solve problem of singular matrix
        cov4=np.cov(class4.T)+np.eye(class4.shape[1])*constant#get covariance and solve problem of singular matrix
    psum=(class0.shape[0]+class1.shape[0]+class2.shape[0]+class3.shape[0]+class4.shape[0])#calculate N
    pc0=class0.shape[0]/psum#calculate P(Ck)
    pc1=class1.shape[0]/psum#calculate P(Ck)
    pc2=class2.shape[0]/psum#calculate P(Ck)
    pc3=class3.shape[0]/psum#calculate P(Ck)
    pc4=class4.shape[0]/psum#calculate P(Ck)
    PG_predicted=np.zeros((X_valid.shape[0],1))#initialization
    for i in range(X_valid.shape[0]):
        y0 = multivariate_normal.logpdf(X_valid[i,:], mu0, cov0)#calculate prior
        y1 = multivariate_normal.logpdf(X_valid[i,:], mu1, cov1)#calculate prior
        y2 = multivariate_normal.logpdf(X_valid[i,:], mu2, cov2)#calculate prior
        y3 = multivariate_normal.logpdf(X_valid[i,:], mu3, cov3)#calculate prior
        y4 = multivariate_normal.logpdf(X_valid[i,:], mu4, cov4)#calculate prior
        # pall = y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4 #P(x)
        pos0 = y0+math.log(pc0)#calculate posterior because P(x) is same so we omit it
        pos1 = y1+math.log(pc1)#calculate posterior because P(x) is same so we omit it
        pos2 = y2+math.log(pc2)#calculate posterior because P(x) is same so we omit it
        pos3 = y3+math.log(pc3)#calculate posterior because P(x) is same so we omit it
        pos4 = y4+math.log(pc4)#calculate posterior because P(x) is same so we omit it
        a={1:pos0,2:pos1,3:pos2,4:pos3,5:pos4}#get dictionary of classes
        PG_predicted[i]=max(a,key=a.get)#get classes responding to max posterior
    if diagonal == 1:
        accuracy_PGdiag = accuracy_score(label_valid, PG_predicted)#compare and get score
        print('\nThe accuracy of Probabilistic Generative classifier HS with diagonal covariance is: ', accuracy_PGdiag * 100, '%')
    else:
        accuracy_PG = accuracy_score(label_valid, PG_predicted)#compare and get score
        print('\nThe accuracy of Probabilistic Generative classifier HS with full covariance is: ', accuracy_PG * 100, '%', 'M is: ', M)
    return PG_predicted

def train7DdataPGfull(Train_7D,labels_7D,diagonal=False):
    #define function train 7D data
    Train = Train_7D#get train data
    labels = labels_7D#get label
    Classes = np.sort(np.unique(labels))#get class
    M = 20
    from sklearn.model_selection import train_test_split
    X_train_class = []#initialize
    for j in range(Classes.shape[0]):
        #classify according to label
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]#get class0
    class1=X_train_class[1]#get class1
    mu0=np.mean(class0,axis=0)#get mean of class0
    mu1=np.mean(class1,axis=0)#get mean of class1
    if diagonal==1:
        cov0 = np.cov(class0.T)*np.eye(class0.shape[1])#get diagonal matrix
        cov1 = np.cov(class1.T)*np.eye(class1.shape[1])#get diagonal matrix
    else:
        cov0=np.cov(class0.T)#get covariance
        cov1=np.cov(class1.T)#get covariance
    pc0=class0.shape[0]/(class0.shape[0]+class1.shape[0])#get p(Ck)
    pc1=class1.shape[0]/(class0.shape[0]+class1.shape[0])#get p(Ck)
    PG_predicted=np.zeros((X_valid.shape[0],1))#initialization
    for i in range(X_valid.shape[0]):
        y0=multivariate_normal.pdf(X_valid[i,:],mu0,cov0)#get prior
        y1=multivariate_normal.pdf(X_valid[i,:],mu1,cov1)#get prior
        pos0=(y0*pc0)/(y0*pc0+y1*pc1)#get posterior
        pos1=(y1*pc1)/(y0*pc0+y1*pc1)#get posterior
        if pos0<pos1:
            #label data according to posterior
            PG_predicted[i]=1
    if diagonal==1:
        accuracy_PGdiag = accuracy_score(label_valid, PG_predicted)
        print('\nThe accuracy of Probabilistic Generative classifier 7D with diagonal covariance is: ', accuracy_PGdiag * 100, '%')
    else:
        accuracy_PG = accuracy_score(label_valid, PG_predicted)
        print('\nThe accuracy of Probabilistic Generative classifier 7D with full covariance is: ', accuracy_PG*100, '%')
    return PG_predicted

def train2DdataPGfull(Train_2D,labels_2D,diagonal=False):
    Train = Train_2D
    labels = labels_2D
    Classes = np.sort(np.unique(labels))
    M = 25
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
    X_train_class = []
    for j in range(Classes.shape[0]):
        # classify according to label
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]#get class0
    class1=X_train_class[1]#get class1
    mu0=np.mean(class0,axis=0)#get mean
    mu1=np.mean(class1,axis=0)#get mean
    if diagonal==1:
        cov0 = np.cov(class0.T)*np.eye(class0.shape[1])#get diagonal matrix
        cov1 = np.cov(class1.T)*np.eye(class1.shape[1])#get diagonal matrix
    else:
        cov0=np.cov(class0.T)#get full covariance
        cov1=np.cov(class1.T)#get full covariance
    pc0=class0.shape[0]/(class0.shape[0]+class1.shape[0])#get pck
    pc1=class1.shape[0]/(class0.shape[0]+class1.shape[0])#get pck
    PG_predicted=np.zeros((X_valid.shape[0],1))
    for i in range(X_valid.shape[0]):
        y0=multivariate_normal.pdf(X_valid[i,:],mu0,cov0)#get prior
        y1=multivariate_normal.pdf(X_valid[i,:],mu1,cov1)#get prior
        pos0=(y0*pc0)/(y0*pc0+y1*pc1)#get posterior
        pos1=(y1*pc1)/(y0*pc0+y1*pc1)#get posterior
        if pos0<pos1:
            PG_predicted[i]=1
    if diagonal==1:
        accuracy_PGdiag = accuracy_score(label_valid, PG_predicted)
        print('\nThe accuracy of Probabilistic Generative classifier 2D with diagonal covariance is: ', accuracy_PGdiag * 100, '%')
    else:
        accuracy_PG = accuracy_score(label_valid, PG_predicted)
        print('\nThe accuracy of Probabilistic Generative classifier 2D with full covariance is: ', accuracy_PG*100, '%')
    return PG_predicted

def TestHSdataPGfull(Train_HS,labels_HS,Test_HS):
    #define a function to train HS data
    Train = Train_HS#get Train data
    labels = labels_HS#get Train laabel
    Classes = np.sort(np.unique(labels))#get class
    M = 15
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
    X_train_class = []#initialize classification
    for j in range(Classes.shape[0]):
        #classify train data according to label
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]#get class
    class1=X_train_class[1]#get class
    class2=X_train_class[2]#get class
    class3=X_train_class[3]#get class
    class4=X_train_class[4]#get class
    mu0=np.mean(class0,axis=0)#get mean
    mu1=np.mean(class1,axis=0)#get mean
    mu2=np.mean(class2,axis=0)#get mean
    mu3=np.mean(class3,axis=0)#get mean
    mu4=np.mean(class4,axis=0)#get mean
    constant=1e-1
    cov0=np.cov(class0.T)+np.eye(class0.shape[1])*constant#get covariance and solve problem of singular matrix
    cov1=np.cov(class1.T)+np.eye(class1.shape[1])*constant#get covariance and solve problem of singular matrix
    cov2=np.cov(class2.T)+np.eye(class2.shape[1])*constant#get covariance and solve problem of singular matrix
    cov3=np.cov(class3.T)+np.eye(class3.shape[1])*constant#get covariance and solve problem of singular matrix
    cov4=np.cov(class4.T)+np.eye(class4.shape[1])*constant#get covariance and solve problem of singular matrix
    psum=(class0.shape[0]+class1.shape[0]+class2.shape[0]+class3.shape[0]+class4.shape[0])#calculate N
    pc0=class0.shape[0]/psum#calculate P(Ck)
    pc1=class1.shape[0]/psum#calculate P(Ck)
    pc2=class2.shape[0]/psum#calculate P(Ck)
    pc3=class3.shape[0]/psum#calculate P(Ck)
    pc4=class4.shape[0]/psum#calculate P(Ck)
    PG_predicted=np.zeros((Test_HS.shape[0],1))#initialization
    for i in range(Test_HS.shape[0]):
        y0 = multivariate_normal.logpdf(Test_HS[i,:], mu0, cov0)#calculate prior
        y1 = multivariate_normal.logpdf(Test_HS[i,:], mu1, cov1)#calculate prior
        y2 = multivariate_normal.logpdf(Test_HS[i,:], mu2, cov2)#calculate prior
        y3 = multivariate_normal.logpdf(Test_HS[i,:], mu3, cov3)#calculate prior
        y4 = multivariate_normal.logpdf(Test_HS[i,:], mu4, cov4)#calculate prior
        # pall = y0 * pc0 + y1 * pc1 + y2 * pc2 + y3 * pc3 + y4 * pc4 #P(x)
        pos0 = y0+math.log(pc0)#calculate posterior because P(x) is same so we omit it
        pos1 = y1+math.log(pc1)#calculate posterior because P(x) is same so we omit it
        pos2 = y2+math.log(pc2)#calculate posterior because P(x) is same so we omit it
        pos3 = y3+math.log(pc3)#calculate posterior because P(x) is same so we omit it
        pos4 = y4+math.log(pc4)#calculate posterior because P(x) is same so we omit it
        a={1:pos0,2:pos1,3:pos2,4:pos3,5:pos4}#get dictionary of classes
        PG_predicted[i]=max(a,key=a.get)#get classes responding to max posterior
    return PG_predicted#return label

def Test7DdataPGfull(Train_7D,labels_7D,Test_7D):
    #define function train 7D data
    Train = Train_7D#get train data
    labels = labels_7D#get label
    Classes = np.sort(np.unique(labels))#get class
    M = 20
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
    X_train_class = []#initialize
    for j in range(Classes.shape[0]):
        #classify according to label
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]#get class0
    class1=X_train_class[1]#get class1
    mu0=np.mean(class0,axis=0)#get mean of class0
    mu1=np.mean(class1,axis=0)#get mean of class1
    cov0=np.cov(class0.T)#get covariance
    cov1=np.cov(class1.T)#get covariance
    pc0=class0.shape[0]/(class0.shape[0]+class1.shape[0])#get p(Ck)
    pc1=class1.shape[0]/(class0.shape[0]+class1.shape[0])#get p(Ck)
    PG_predicted=np.zeros((Test_7D.shape[0],1))#initialization
    for i in range(Test_7D.shape[0]):
        y0=multivariate_normal.pdf(Test_7D[i,:],mu0,cov0)#get prior
        y1=multivariate_normal.pdf(Test_7D[i,:],mu1,cov1)#get prior
        pos0=(y0*pc0)/(y0*pc0+y1*pc1)#get posterior
        pos1=(y1*pc1)/(y0*pc0+y1*pc1)#get posterior
        if pos0<pos1:
            #label data according to posterior
            PG_predicted[i]=1
    return PG_predicted

def Test2DdataPGfull(Train_2D,labels_2D,Test_2D):
    #same with above 7D and HS
    Train = Train_2D
    labels = labels_2D
    Classes = np.sort(np.unique(labels))
    M = 25
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
    X_train_class = []
    for j in range(Classes.shape[0]):
        jth_class = X_train[label_train == Classes[j],:]
        X_train_class.append(jth_class)
    class0=X_train_class[0]
    class1=X_train_class[1]
    mu0=np.mean(class0,axis=0)
    mu1=np.mean(class1,axis=0)
    cov0=np.cov(class0.T)
    cov1=np.cov(class1.T)
    pc0=class0.shape[0]/(class0.shape[0]+class1.shape[0])
    pc1=class1.shape[0]/(class0.shape[0]+class1.shape[0])
    PG_predicted=np.zeros((Test_2D.shape[0],1))
    for i in range(Test_2D.shape[0]):
        y0=multivariate_normal.pdf(Test_2D[i,:],mu0,cov0)
        y1=multivariate_normal.pdf(Test_2D[i,:],mu1,cov1)
        pos0=(y0*pc0)/(y0*pc0+y1*pc1)
        pos1=(y1*pc1)/(y0*pc0+y1*pc1)
        if pos0<pos1:
            PG_predicted[i]=1
    return PG_predicted
"""
===============================================================================
===============================================================================
============================ KNN Classifier ===================================
===============================================================================
===============================================================================
"""

""" Here you can write functions to achieve your KNN classifier. """
#Here define KNN, we use it to train and get cross validation
def KNN(Train,labels,k):
    def KNNClassifer(train,test,labels,k):
        distance = np.sum((test-train)**2,axis=1)**0.5#calculate distance
        classcount={}#initialization dictionary
        sortdistance=distance.argsort()#sort according to distance from small to large
        for i in range(k):
            #Here we use first k elements
            votelabel=labels[sortdistance[i]]#get label through the location of first k smallest distance
            classcount[votelabel]=classcount.get(votelabel,0)+1#calculate times of each label
        label=collections.Counter(classcount).most_common(1)[0][0]#get the key of most value and get label
        return label
    M = 26
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size=0.33, random_state=M)

    predictions_KNN=np.zeros((X_valid.shape[0],1))
    for i in range(X_valid.shape[0]):
        predictions_KNN[i] = KNNClassifer(X_train,X_valid[i,:],label_train,k)#get cross_validation
    accuracy_KNN = accuracy_score(label_valid, predictions_KNN)
    print('\nThe accuracy of KNN classifier is: ', accuracy_KNN * 100, '%',' k is: ',k)
# #k vary from samll to large
# for k in range(24):
#     KNN(Train_HS, labels_HS, k+1)

#get test data's label using KNN
def TestKNN(Train,labels,Test_HS,k):
    def KNNClassifer(train,test,labels,k):
        distance = np.sum((test-train)**2,axis=1)**0.5#calculate distance
        classcount={}#initialization dictionary
        sortdistance=distance.argsort()#sort according to distance from small to large
        for i in range(k):
            #Here we use first k elements
            votelabel=labels[sortdistance[i]]#get label through the location of first k smallest distance
            classcount[votelabel]=classcount.get(votelabel,0)+1#calculate times of each label
        label=collections.Counter(classcount).most_common(1)[0][0]#get the key of most value and get label
        return label
    M = 26
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size=0.33, random_state=M)

    predictions_KNN=np.zeros((Test_HS.shape[0],1))
    for i in range(Test_HS.shape[0]):
        predictions_KNN[i] = KNNClassifer(X_train,Test_HS[i,:],label_train,k)#get label of each row
    return predictions_KNN#get label

""" ============  Generate Training and validation Data =================== """
k=13#initialize the best k
trainHSdataPGfull(Train_HS,labels_HS,diagonal=False)#cross validation with full covariance
trainHSdataPGfull(Train_HS,labels_HS,diagonal=True)#cross validation with diagonal covariance
KNN(Train_HS,labels_HS,k)#cross validation for KNN
train7DdataPGfull(Train_7D,labels_7D,diagonal=False)#cross validation with full covariance
train7DdataPGfull(Train_7D,labels_7D,diagonal=True)#cross validation with diagonal covariance
KNN(Train_7D,labels_7D,k)#cross validation for KNN
train2DdataPGfull(Train_2D,labels_2D,diagonal=False)#cross validation with full covariance
train2DdataPGfull(Train_2D,labels_2D,diagonal=True)#cross validation with diagonal covariance
KNN(Train_2D,labels_2D,k)#cross validation for KNN






""" ========================  Test the Model ============================== """

""" This is where you should test the testing data with your classifier """

TestLabel2D=Test2DdataPGfull(Train_2D,labels_2D,Test_2D)#get label
TestLabel7D=Test7DdataPGfull(Train_7D,labels_7D,Test_7D)#get label
TestLabelHS=TestHSdataPGfull(Train_HS,labels_HS,Test_HS)#get label

np.savetxt('2DforTestLabels.txt',TestLabel2D)#save as txt
np.savetxt('7DforTestLabels.txt',TestLabel7D)#save as txt
np.savetxt('HSforTestLabels.txt',TestLabelHS)#save as txt