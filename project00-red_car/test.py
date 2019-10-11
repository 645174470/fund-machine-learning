
""" =======================  Import dependencies ========================== """

import train
from sklearn import neighbors
import numpy as np

""" =======================  Get parameters from train.py ========================== """
knn = train.KNN()
# Get threshold parameters
parameters = train.trainHSV()
hmin = parameters[0]
hmax = parameters[1]
smin = parameters[2]
vmin = parameters[3]

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
def calculatedistance(A, B):

    ''' Here is a fast calculation formula of distance of all rows, I get a new N*N matrix and the i row j column
    responding to distance between i row and j row from original matrix '''

    BT = B.T  # transpose
    innerproduct = A@BT  # inner product
    squareA = np.sum(A**2, axis=1)  # square and sum elements at same row
    transsqA = squareA.reshape(innerproduct.shape[1], 1)  # equals to transpose
    SquareA = np.tile(transsqA, (1, innerproduct.shape[1]))  # copy to N column
    squareB = np.sum(B**2, axis=1)  # square and sum elements at same row
    SquareB = np.tile(squareB, (innerproduct.shape[0], 1))  # copy to N row
    dist = (SquareA + SquareB - 2*innerproduct)**0.5  # create a shape of square of minus then get distance
    return dist

def KNNerrorkiller(testset, testdata):
    # I initialized KNN's parameter below and this is only a predict part
    testresult = []
    for i in range(testset.shape[0]):
        outcome = knn.predict([RGB2HSV(testdata[testset[i, 0], testset[i, 1], :])])  # judge RGB value to label
        if outcome == 0:
            testresult.append(testset[i, :])  # return location in data_train
    testresult = np.array(testresult)
    return testresult

def predict(yourtestpicture):
    # This function is used to predict red cars' locations
    c = []
    # Firstly, I convert all RGB value to HSV and choose those meet our requirements and return their locations
    for i in range(yourtestpicture.shape[0]):
        for j in range(yourtestpicture.shape[1]):
            if hmin <= RGB2HSV(yourtestpicture[i, j, :])[0] <= hmax \
                    and RGB2HSV(yourtestpicture[i, j, :])[1] >= smin \
                    and RGB2HSV(yourtestpicture[i, j, :])[2] >= vmin:
                c.append([i, j])  # get location
    redpixel = np.array(c)
    # np.save('redpixel.npy', redpixel)

    '''Here is a judgment to avoid memory overflow. If the matrix is huge, I have to shrink it. Considering a red
       car's pixel is continuous, it will influence slightly if I remove some row. Here I choose to remove all even
       rows if overflow'''

    testresult = KNNerrorkiller(redpixel, yourtestpicture)  # Use KNN to remove interference points
    # np.save('KNNresult.npy',testresult)
    if testresult.shape[0] > 20000:
        mode = [x for x in range(testresult.shape[0]) if x % 2 == 0]  # get a list which contains even numbers
        deletemode = np.array(mode)
        testresult = np.delete(testresult, deletemode, 0)  # delete even rows to avoid memory overflow

    close = []
    count = calculatedistance(testresult, testresult)  # use fast calculation formula but may cause memory overflow

    '''Considering the fact, our pixels are continuous, so we choose to remove closed points'''

    distancethreshold = 10
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            if count[i, j] < distancethreshold and i < j:
                close.append([i, j])  # get close rows
    closepoint = np.array(close)
    # np.save('closedata.npy', closepoint)
    '''If my i, j occurs, it shows they are closed to some point, we use this way to get out some noise red pixel'''

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

    '''Here return to your picture and get locations of our predicted points'''

    outcome = []
    for i in range(uniall.shape[0]):
        outcome.append(testresult[uniall[i], :])
    redcarlocation = np.array(outcome)
    return redcarlocation