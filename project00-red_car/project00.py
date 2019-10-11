"""
File:        project00.py
Author:      Hao Sun
Description: The main method of this code has three parts. Firstly, I convert RGB to HSV and according
             to ground_truth to train and find some "red pixel". Then I find in fact the result contains
             some undesired elements such as orange ground and roofs, of course and red roofs. I use KNN
             to eliminate orange pixel. Finally, find points which have close red pixel and remove the
             close point. Then I get the location of red car. The whole project to test two full picture
             takes about 1000 seconds.
"""
""" =======================  Import dependencies ========================== """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import time

""" =======================  Import DataSet ========================== """
start = time.clock()
data_train = np.load("data_train.npy")
ground_truth = np.load("ground_truth.npy")
data_test = np.load("data_test.npy")
error_pixel = np.load('error_pixel.npy') # This file is created by myself. It contains information about 'not a red car'
red_car = np.load('red_car.npy')  # This file is created by myself. Additional red car information.


""" ======================  Function definitions ========================== """

def RGB2HSV(pixel):
    # use this function to convert RGB value to HSV space
    v = max (pixel)
    vn = min (pixel)
    r = int(pixel[0])   # get r value,use int to avoid warning of overflow 255
    g = int(pixel[1])   # get g value,use int to avoid warning of overflow 255
    b = int(pixel[2])   # get b value,use int to avoid warning of overflow 255
    # below are transform formula
    if v == vn:
        h = 0
    elif r == v:
        h = 60 * (g-b)/(v-vn)
    elif g == v:
        h = 120 + 60 * (b-r)/(v-vn)
    elif b == v:
        h = 240 + 60 * (r-g)/(v-vn)
    if h > 180:
        h = h - 360
    if v == vn:
        s=0
    else:
        s=(v-vn)/v*255
    return np.array([h, s, v])
def calculatedistance(A, B):
    '''
    Here is a fast calculation formula of distance of all rows, I get a new N*N matrix and the i row j column
    responding to distance between i row and j row from original matrix
    '''
    BT = B.T    # transpose
    innerproduct = A@BT  # innner product
    squareA = np.sum(A**2, axis=1)  # square and sum elements at same row
    transsqA = squareA.reshape(innerproduct.shape[1], 1)    # equals to transpose
    SquareA = np.tile(transsqA, (1, innerproduct.shape[1]))     # copy to N column
    squareB = np.sum(B**2, axis=1)      # square and sum elements at same row
    SquareB = np.tile(squareB, (innerproduct.shape[0], 1))  # copy to N row
    dist = (SquareA + SquareB - 2*innerproduct)**0.5    # create a shape of square of minus then get distance
    return dist
def KNNerrorkiller(testset, data_train):
    # I initialized KNN's parameter below and this is only a predict part
    testresult = []
    for i in range(testset.shape[0]):
            outcome = knn.predict([RGB2HSV(data_train[testset[i, 0], testset[i, 1], :])])  # judge RGB value to label
            if outcome == 0:
                testresult.append(testset[i, :])  # return location in data_train
    testresult = np.array(testresult)
    return testresult
def predict(data_train, hmin, hmax, smin, vmin):
    # This function is used to predict red cars' locations
    c = []
    # Firstly, I convert all RGB value to HSV and choose those meet our requirements and return their locations
    for i in range(data_train.shape[0]):
        for j in range(data_train.shape[1]):
            if hmin <= RGB2HSV(data_train[i, j, :])[0] <= hmax \
                    and RGB2HSV(data_train[i, j, :])[1] >= smin \
                    and RGB2HSV(data_train[i, j, :])[2] >= vmin:
                c.append([i, j])    # get location
    redpixel = np.array(c)

    '''Here is a judgment to avoid memory overflow. If the matrix is huge, I have to shrink it. Considering a red
       car's pixel is continuous, it will influence slightly if I remove some row. Here I choose to remove all even
       rows if overflow'''

    testresult = KNNerrorkiller(redpixel, data_train)   # Use KNN to remove interference points
    if testresult.shape[0] > 20000:
        mode = [x for x in range(testresult.shape[0]) if x % 2 == 0]    # get a list which contains even numbers
        deletemode = np.array(mode)
        testresult = np.delete(testresult, deletemode, 0)   # delete even rows to avoid memory overflow

    close = []
    count = calculatedistance(testresult, testresult)   # use fast calculation formula but may cause memory overflow

    '''Considering the fact, our pixles are continuous, so we choose to remove close points'''

    distancethreshold = 8
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            if count[i, j] < distancethreshold and i < j:
                close.append([i, j])    # get close rows
    closepoint = np.array(close)

    '''If my i, j occurs, it shows they are close to some point, we use this way to get out some noise red pixel'''

    uni1 = np.unique(closepoint[:, 1])  # get unique number of rows
    uni0 = np.unique(closepoint[:, 0])  # get unique number of rows

    '''Judgment repeated rows exists in i and j simultaneously and remove repeated rows'''

    delete = []
    for i in range(uni0.shape[0]):
        for j in range(uni1.shape[0]):
            if uni0[i] == uni1[j]:
                delete.append(i)
    de = np.array(delete)
    uniall = np.delete(uni0, de, 0)

    '''Here return to data_train and get location of our predicted points'''

    outcome = []
    for i in range(uniall.shape[0]):
        outcome.append(testresult[uniall[i], :])
    redcarlocation = np.array(outcome)
    return redcarlocation
    # return testresult

""" ======================  Get Train dataset  ========================== """


a = []  # initialization
# Here I train the data using ground_truth
for i in range(ground_truth.shape[0]):
    a.append(RGB2HSV(data_train[ground_truth[i, 1], ground_truth[i, 0], :]))    # get HSV value
b = np.array(a)


"""
=======================================================================================
============================ Train and Cross-Validation ===============================
=======================================================================================

Here I choose 3 1-D Gaussian Distributions according to the characteristic of HSV space.
H(hue) is a angel which decides color, S(Saturation) is amount of grey and V(Value)
is brightness. So theoretically a red pixel is around some h value and also have a 
relatively high value in S and V to except grey and black.
"""
Train = b
labels = np.ones((Train.shape[0], 1))   # initialize label
M = 26  # This value is chosen by circulation of different M
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size=0.33, random_state=M)
mean = np.mean(X_train, axis=0) # calculate mean
cov = np.diag(np.diag(np.cov(X_train.T)))
std = np.sum(cov, axis=0)**0.5  # calculate std
prediction = []

'''Here I initialize parameters. Because pdf is hard to decide a threshold, I calculate Q function of Gaussian 
Distribution and find if a point is far from mean, its probability is small, in other word, it doesn't belong to
my Gaussian. And I change this threshold  according to its effect. Finally, I think this threshold is effective'''
threshold = 4
hmin = mean[0] - std[0] # regular range
hmax = mean[0] + std[0] # regular range
smin = mean[1] - std[1]/threshold   # regular range
vmin = mean[2] - std[2]/threshold   # regular range
for i in range(X_valid.shape[0]):
    if hmin <= X_valid[i, 0] <= hmax and X_valid[i, 1] >= smin and X_valid[i, 2] >= vmin:
        prediction.append(1)
    else:
        prediction.append(0)
accuracy = accuracy_score(label_valid, prediction)
print('\nThe accuracy is: ', accuracy*100, '%''M is ', M)

'''Initialize KNN parameter'''
n_neighbors = 10
truevalue = []
for i in range(ground_truth.shape[0]):
    truevalue.append(data_train[ground_truth[i, 1], ground_truth[i, 0], :])  # get RGB value
truevalue = np.array(truevalue)
trainset = np.vstack((truevalue, red_car, error_pixel))     # joint my train data with ground_truth
transtrainset = []
for i in range(trainset.shape[0]):
    transtrainset.append(RGB2HSV(trainset[i,:]))
transtrainset = np.array(transtrainset)
label0 = np.zeros(((ground_truth.shape[0]+red_car.shape[0]), 1))
label1 = np.ones((error_pixel.shape[0], 1))
label = np.vstack((label0, label1))     # initialize label
knn = neighbors.KNeighborsClassifier(n_neighbors)   # use KNN
knn.fit(transtrainset, label.ravel())   # fit data


""" ========================  Test the Model ============================== """

'''Test module using predict function and get result of two pictures'''


redcarlocation = predict(data_train, hmin, hmax, smin, vmin)
testresult = predict(data_test, hmin, hmax, smin, vmin)
plt.figure(1)
plt.imshow(data_train)
plt.scatter(redcarlocation[:, 1], redcarlocation[:, 0])
plt.figure(2)
plt.imshow(data_test)
plt.scatter(testresult[:, 1], testresult[:, 0])
end = time.clock()
times = end-start
print(times)