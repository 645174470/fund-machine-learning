import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2 as cv
import time

'''load data'''
trainimage = np.load('TrainImages.npy')
classdata = np.load('ClassData.npy')
classlabel = np.load('ClassLabels.npy')
trainy = np.load('TrainY.npy')


#t_start = time.clock()
# def picreshape(data):
#     for i in range(data.shape[0]):
#         data[i] = cv.resize(data[i], (60, 60), interpolation=cv.INTER_CUBIC)
#     return data
'''define function'''
normal_size = 60
# def Normalize(data,size = normal_size):
#     # Normalize single data to 60*60
#     [rows, cols] = data.shape
#     if rows < size:
#         data = np.vstack([data, np.zeros([size - rows, cols])])
#     elif rows > size:
#         b = np.zeros((int(data.shape[0] / 2), data.shape[1]))
#         for i in range(int(data.shape[0] / 2)):
#             b[i, :] = data[2 * i, :]
#         rows = b.shape[0]
#         data = np.vstack([b, np.zeros([size - rows, cols])])
#     if cols < size:
#         data = np.hstack([data, np.zeros([size, size - cols])])
#     elif cols > size:
#         b = np.zeros((data.shape[0], int(data.shape[1]/2)))
#         for i in range(int(data.shape[1] / 2)):
#             b[:, i] = data[:, 2 * i]
#         bcols = b.shape[1]
#         data = np.hstack([b, np.zeros([size, size - bcols])])
#     return data
def picreshape(data):
    for i in range(data.shape[0]):
        data[i] = cv.resize(data[i], (60, 60), interpolation=cv.INTER_CUBIC)
    return data
#
# def datapreprocess(data):
#     # preprocess all dataset
#     tmp = []
#     for i in range(data.shape[0]):
#         tmp.append(Normalize(data[i]))
#     return tmp
classdata = np.hstack([trainimage,classdata])
trainy=trainy.transpose()[0]
classlabel=np.hstack([trainy,classlabel])
classdata = picreshape(classdata)
tmp=[]
for i in range(classdata.shape[0]):
    tmp.append(classdata[i])
classdata=tmp
M = 15
X_train, X_valid, target_train, target_valid \
    = train_test_split(classdata, classlabel, test_size=0.33, random_state=M)  # cross validation

X_train = torch.FloatTensor(X_train)
X_train = torch.unsqueeze(X_train,dim=1).type(torch.FloatTensor)
X_valid = torch.FloatTensor(X_valid)
X_valid = torch.unsqueeze(X_valid, dim=1).type(torch.FloatTensor)
target_train = torch.from_numpy(target_train)
# target_valid = torch.from_numpy(target_valid)
# trainimage = datapreprocess(trainimage)
'''Initialize CNN parameters'''
EPOCH = 20
batch_size = 50
eta = 0.001
kernel_size = 7
pool_size1 = 4
pool_size2 = 4

'''Create CNN train dataset'''
train_dataset = Data.TensorDataset(X_train, target_train)

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)


'''CNN'''
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


cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=eta)   # optimize cnn parameters using Adam
#optimizer = torch.optim.SGD(cnn.parameters(), lr=eta)
loss_func = nn.CrossEntropyLoss()   # loss function


# training and testing
for epoch in range(EPOCH):
    for i, (X_train, target_train) in enumerate(train_loader):
        output = cnn(X_train)               # cnn output
        loss = loss_func(output, target_train)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if i % 50 == 0:
            cnn.eval()
            test_output = cnn(X_valid)
            predict = torch.max(test_output, 1)[1].data.numpy()
            cnn.train()
            accuracy = accuracy_score(predict, target_valid)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#t_end = time.clock()
#print('Time is :',(t_end-t_start))
