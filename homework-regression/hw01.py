# -*- coding: utf-8 -*-
"""
File:   hw01.py
Author: 
Date:   
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
	'''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
	# x = np.random.uniform(l,u,N)
	step = (u-l)/(N);
	x = np.arange(l+step/2,u+step/2,step)
	e = np.random.normal(0,gVar,N)
	t = np.sinc(x) + e
	return x,t

def plotData2(x1,t1,x2,t2,x3,t3,x4,t4,M,legend=[]):
    #Use this function to show test, train, eatimated function in different M
    p1 = plt.plot(x1, t1, 'go') #plot test data
    p2 = plt.plot(x2, t2, 'bo') #plot train data
   
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    plt.title('M='+str(M)+', N='+str(N))
    p3=plt.plot(x3,t3,'r')#plot estimated function
    p4=plt.plot(x4,t4,'black')#plot true function
    

    plt.legend((p1[0],p2[0],p3[0],p4[0]),legend)
        
    """
    This seems like a good place to write a function to learn your regression
    weights!
    
    """
def fitdata(x,t,M):
	#Through fitdata to get w
    X=np.array([x**m for m in range(M+1)]).T
    w=np.linalg.inv(X.T@X)@X.T@t
    return w
        

""" ======================  Variable Declaration ========================== """

l = 0 #lower bound on x
u = 10 #upper bound on x
N = 10 #number of samples to generate
gVar = .25 #variance of error distribution
#M =  3 #regression model order
""" =======================  Generate Training Data ======================= """
data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T

x1 = data_uniform[:,0]
t1 = data_uniform[:,1]#true value of train data

x2 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
t2 = np.sinc(x2) #compute the true function value of training data
    
""" ========================  Train the Model ============================= """




#plotData(x1,t1,x2,t2,x3,t3,['Training Data', 'True Function', 'Estimated\nPolynomial'])


""" ======================== Generate Test Data =========================== """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """
   
def generateTestData(N,l,u,gVart):
	#generate TestData which has the same mean and variance with train data
    step=(u-l)/N
    x=np.arange(l+step/2,u+step/2,step)
    e=np.random.normal(0,gVart,N)
    t=np.sinc(x)+e
    return x,t
Nt=100#number of test data
data_test=np.array(generateTestData(Nt,l,u,gVar)).T#genetate test data
xt=data_test[:,0]
tt=data_test[:,1]#ture value of test data

""" ========================  Test the Model ============================== """


""" This is where you should test the validation set with the trained model """
plt.figure(1)
q=12
EteR=np.zeros(q)#initialize Erms for test data
EtrR=np.zeros(q)#initialize Erms for training data
Q=range(q)#generate M
for M in Q:
	#Use "for" to get different Erms for test data and Erms for training data corresponding to M
    w = fitdata(x1,t1,M) #get w of training data
    X = np.array([x1**m for m in range(w.size)]).T
    t3 = X@w #compute the predicted value of training data
    Xt = np.array([xt**m for m in range(w.size)]).T
    t4 = Xt@w#compute the predicted value of test data
    EteR[M]=(((t4-tt)@(t4-tt).T)/Nt)**(1/2)#calculate Erms of test data
    EtrR[M]=(((t3-t1)@(t3-t1).T)/N)**(1/2)#calculate Erms of training data
    plt.subplot(3,4,M+1)
    plotData2(xt,tt,x1,t1,x1,t3,x2,t2,M,legend=['test data','train data','estimated function','ture value'])
    #show relationship of test, train, eatimated function, true value
def plotData(x1,t1,x2,t2,legend=[]):
	#plot figure of M and Erms of test data and Erms of training data
    p1 = plt.plot(x1, t1, 'bo-') #plot training Erms
    p2 = plt.plot(x2, t2, 'go-') #plot test Erms
    plt.ylabel(r'$\ E_(RMS) $',fontsize=20) #label y axes
    plt.xlabel('M',fontsize=20)#label x axes
    plt.legend((p1[0],p2[0]),legend,fontsize=20)
    plt.tick_params(labelsize=20)
plt.figure(2)
plotData(Q,EtrR,Q,EteR,['Training data', 'Test data'])
#plot Erms of training data and test data
#plt.show()

"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """

#True distribution mean and variance 
trueMu = 4 #u for ML
trueVar = 2 #Var for ML

#Initial prior distribution mean and variance (You should change these parameters to see how they affect the ML and MAP solutions)
priorMu = 8
priorVar = .2

numDraws = 200 #Number of draws from the true distribution


"""========================== Plot the true distribution =================="""
#plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
#plt.figure(0)
#p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
#plt.title('Known "True" Distribution')

"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws

drawResult = []
ML=[]
MAP=[]
for draw in range(numDraws):
    drawResult.append(np.random.normal(trueMu,trueVar,1)[0])#draw data from Gaussian Distribution
    N=len(drawResult)
    uML=sum(drawResult)/N#calculate maximum likelihood solution for µ given the N data points
    uN=trueVar/(N*priorVar+trueVar)*priorMu+N*priorVar/(N*priorVar+trueVar)*uML#calculate MAP solution for µ given the N data points
    NVar=trueVar*priorVar/(trueVar+priorVar*N)#calculate MAP solution for variance given the N data points
    #print(drawResult)
    #print('Frequentist/Maximum Likelihood Probability:' + str(uML))
    #print('Bayesian/MAP Probability:' + str(uN))
    #input("Hit enter to continue...\n")
    ML.append(uML)#get a list of uML
    MAP.append(uN)#get a list of uMAP
    priorMu=uN#change mean from prior distribution to posterior distribution
    priorVar=NVar#change variance from prior distribution to posterior distribution
"""
You should add some code to visualize how the ML and MAP estimates change
with varying parameters.  Maybe over time?  There are many differnt things you could do!
"""
def plotDrawData(x1,t1,x2,t2,legend=[]):
    #plot figure of M and Erms of test data and Erms of training data
    p1 = plt.plot(x1, t1, 'bo-') #plot uML
    p2 = plt.plot(x2, t2, 'ro-') #plot uMAP
    plt.ylabel(r'$\mu $',fontsize=20) #label y axes
    plt.xlabel( 'Number of Draws',fontsize=20)#label x axes
    plt.legend((p1[0],p2[0]),legend,fontsize=20)
    plt.tick_params(labelsize=20)
D=np.arange(numDraws)
plt.figure(3)
plotDrawData(D,ML,D,MAP,['ML','MAP'])#plot uML and uMAP change with number of draws
#plt.plot(np.arange (numDraws),ML,'bo-')
#plt.plot(np.arange (numDraws),MAP,'ro-')
#plt.show()
