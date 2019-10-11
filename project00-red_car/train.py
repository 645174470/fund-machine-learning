
""" =======================  Import dependencies ========================== """

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
# from sklearn.metrics import accuracy_score

""" =======================  Import DataSet ========================== """

data_train = np.load("data_train.npy")
ground_truth = np.load("ground_truth.npy")
error_pixel=np.load('error_pixel.npy')  # This file is created by myself. It contains imformation about 'not a red car'
red_car=np.load('red_car.npy')  # This file is created by myself. Additional red car information.


""" ======================  Function definitions ========================== """

def RGB2HSV(pixel):
    # use this function to convert RGB value to HSV space
    v = max(pixel)
    vn = min(pixel)
    r = int(pixel[0])  # get r value,use int to avoid warning of overflow 255
    g = int(pixel[1])  # get g value,use int to avoid warning of overflow 255
    b = int(pixel[2])  # get b value,use int to avoid warning of overflow 255
    # below are transform formula
    if v == vn:
        h = 0
    elif r == v:
        h = 60 * (g - b) / (v - vn)
    elif g == v:
        h = 120 + 60 * (b - r) / (v - vn)
    elif b == v:
        h = 240 + 60 * (r - g) / (v - vn)
    if h > 180:
        h = h - 360
    if v == vn:
        s = 0
    else:
        s = (v - vn) / v * 255
    return np.array([h, s, v])
def trainHSV():
    """
    Here I choose 3 1-D Gaussian Distributions according to the characteristic of HSV space.
    H(hue) is a angel which decides color, S(Saturation) is amount of grey and V(Value)
    is brightness. So theoretically a red pixel is around some h value and also have a
    relatively high value in S and V to except grey and black.
    """

    a = []  # initialization
    # Here I train the data using ground_truth
    for i in range(ground_truth.shape[0]):
        a.append(RGB2HSV(data_train[ground_truth[i, 1], ground_truth[i, 0], :]))  # get HSV value
    b = np.array(a)
    Train = b
    labels = np.ones((Train.shape[0], 1))  # initialize label
    M = 26  # This value is chosen by circulation of different M
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size=0.33, random_state=M)
    mean = np.mean(X_train, axis=0)  # calculate mean
    cov = np.diag(np.diag(np.cov(X_train.T)))
    std = np.sum(cov, axis=0)**0.5  # calculate std

    '''Here I initialize parameters. Because pdf is hard to decide a threshold, I calculate Q function of Gaussian 
    Distribution and find if a point is far from mean, its probability is small, in other word, it doesn't belong to
    my Gaussian. And I change this threshold  according to its effect. Finally, I think this threshold is effective'''
    threshold = 4
    hmin = mean[0] - std[0]  # regular range
    hmax = mean[0] + std[0]  # regular range
    smin = mean[1] - std[1]/threshold  # regular range
    vmin = mean[2] - std[2]/threshold  # regular range
    # '''cross validation'''
    # prediction = []
    # for i in range(X_valid.shape[0]):
    #     if hmin <= X_valid[i, 0] <= hmax and X_valid[i, 1] >= smin and X_valid[i, 2] >= vmin:
    #         prediction.append(1)
    #     else:
    #         prediction.append(0)
    # accuracy = accuracy_score(label_valid, prediction)
    # print('\nThe accuracy is: ', accuracy*100, '%''M is ', M)
    return [hmin, hmax, smin, vmin]
def traindataKNN():

    # Initialize KNN parameter

    truevalue = []
    for i in range(ground_truth.shape[0]):
        truevalue.append(data_train[ground_truth[i, 1], ground_truth[i, 0], :])  # get RGB value
    truevalue = np.array(truevalue)
    trainset = np.vstack((truevalue, red_car, error_pixel))  # joint my train data with ground_truth

    # convert to HSV

    transtrainset=[]
    for i in range(trainset.shape[0]):
        transtrainset.append(RGB2HSV(trainset[i,:]))
    transtrainset=np.array(transtrainset)
    return transtrainset
def labelKNN():
    label0 = np.zeros(((ground_truth.shape[0]+red_car.shape[0]), 1))  # label red car
    label1 = np.ones((error_pixel.shape[0], 1))  # label not red car
    label = np.vstack((label0, label1))  # initialize label
    return label
def KNN():
    KNNdata = traindataKNN()
    labelforKNN = labelKNN()
    n_neighbors = 10
    knn = neighbors.KNeighborsClassifier(n_neighbors)  # use KNN
    knn.fit(KNNdata, labelforKNN.ravel())  # fit data
    return knn