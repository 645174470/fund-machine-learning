
import numpy as np
import mlp
from sklearn.model_selection import train_test_split

#load data
data = np.load('dataSet.npy')


#Set up Neural Network
data_in = data[:,:-1]
target_in = data[:,2].reshape(data.shape[0],1)

hidden_layers = 6
NN = mlp.mlp(data_in,target_in,hidden_layers)

#Analyze Neural Network Performance
M=15
X_train, X_valid, target_train, target_valid = train_test_split(data_in, target_in, test_size = 0.33, random_state = M)
eta=0.7
iteration=100
print('iteration is ',iteration)
NN.earlystopping(X_train,target_train,X_valid,target_valid,eta,iteration)
NN.confmat(data_in,target_in)

