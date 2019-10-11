import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data
from sklearn import neighbors
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

trainimage = np.load('TrainImages.npy')
classdata = np.load('ClassData.npy')
classlabel = np.load('ClassLabels.npy')
trainy = np.load('TrainY.npy')

def picreshape(data):
    for i in range(data.shape[0]):
        data[i] = cv.resize(data[i], (60, 60), interpolation=cv.INTER_CUBIC)
    return data

classdata = np.hstack([trainimage,classdata])
classdata = picreshape(classdata)
classdata = classdata.reshape(classdata.shape[0],1)
trainy=trainy.transpose()[0]
classlabel=np.hstack([trainy,classlabel])
normalclassdata = []
for i in range(classdata.shape[0]):
    normalclassdata.append(classdata[i][0].reshape(3600,1))
normalclassdata = np.array(normalclassdata)
normalclassdata = normalclassdata.reshape(normalclassdata.shape[0],normalclassdata.shape[1])
M = 15
X_train, X_valid, target_train, target_valid \
    = train_test_split(normalclassdata, classlabel, test_size=0.33, random_state=M)  # cross validation
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train,target_train)
predict = knn.predict(X_valid)
accuracy = accuracy_score(predict, target_valid)
print('KNN accuracy :',accuracy)