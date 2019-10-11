import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

trainimage = np.load('TrainImages.npy')
classdata = np.load('ClassData.npy')
classlabel = np.load('ClassLabels.npy')
trainy = np.load('TrainY.npy')
indata = np.load('inData.npy')

def picreshape(data):
    # reshape picture to normalize
    for i in range(data.shape[0]):
        data[i] = cv.resize(data[i], (60, 60), interpolation=cv.INTER_CUBIC)
    return data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(       # input shape (1, 60, 60)
            nn.Conv2d(
                in_channels=1,            # input height
                out_channels=18,          # n_filters
                kernel_size=kernel_size,  # filter size
                stride=1,                 # filter step
                padding=3                # keep same size
            ),                            # output shape (18, 60, 60)
            nn.Dropout(0.5),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=pool_size1),  # output shape (18, 15, 15)
        )
        self.conv2 = nn.Sequential(       # input shape (18, 15, 15)
            nn.Conv2d(
                in_channels=18,           # input height
                out_channels=36,          # n_filters
                kernel_size=kernel_size,  # filter size
                stride=1,                 # filter step
                padding=3                 # keep same size
            ),                            # output shape (36, 15, 15)
            nn.Dropout(0.5),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(pool_size2),  # output shape (36, 3, 3)
        )
        scalar = pool_size1*pool_size2
        self.out = nn.Linear(36 * int(normal_size/scalar) * int(normal_size/scalar), 9)  # fully connected layer

    def forward(self, input):
        # forward propagation
        rescov1 = self.conv1(input)
        rescov2 = self.conv2(rescov1)
        rescov2 = rescov2.view(rescov2.size(0), -1)
        output = self.out(rescov2)
        return output

indata = picreshape(indata)
tmp=[]
for i in range(indata.shape[0]):
    tmp.append(indata[i])
indata=tmp
indata = torch.FloatTensor(indata)
indata = torch.unsqueeze(indata, dim=1).type(torch.FloatTensor)
#cnn = train.train_CNN()
cnn = torch.load('cnn.pkl')
cnn.eval()
output = cnn(indata)
predict = torch.max(output, 1)[1].data.numpy()
np.save('out.npy', predict)