# -*- coding: utf-8 -*-
"""
File:   hw04.py
Author: Hao Sun
Date:   
Desc:   
    
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score


dataSet1=np.load('dataSet1.npy')
dataSet2=np.load('dataSet2.npy')
dataSet3=np.load('dataSet3.npy')
dataSet1test = np.delete(dataSet1, 2, axis=1)
dataSet2test = np.delete(dataSet2, 2, axis=1)
dataSet3test = np.delete(dataSet3, 2, axis=1)


def plotLineX(weights, range):
    y = np.array(range)
    x = (weights[0])+(weights[1])*y
    plt.plot(x, y)


def plotLineY(weights, range):
    x = np.array(range)
    y = (weights[0])+(weights[1])*x
    plt.plot(x, y)


def plotfigure1():
    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.scatter(dataSet1[:, 0], dataSet1[:, 1], c=dataSet1[:, 2])
    y_range = [-0.5,2.5]
    plotLineX([0.5,0], y_range)   # x=0.5
    plotLineX([1.5,0], y_range)   # x=1.5
    neuron1 = dataSet1test@[1, 0]-0.5  # x-0.5=n1
    neuron2 = dataSet1test@[1, 0]-1.5  # x-1.5=n2
    # Activation function if v>0 v=1 if v<=0 v=-1
    for i in range(neuron1.shape[0]):
        if neuron1[i] > 0:
            neuron1[i] = 1
        else:
            neuron1[i] = -1
    for i in range(neuron2.shape[0]):
        if neuron2[i] > 0:
            neuron2[i] = 1
        else:
            neuron2[i] = -1
    neuron = 1*((neuron1 - neuron2 - 1) > 0)
    accuracy = accuracy_score(dataSet1[:, 2], neuron)
    print('\nThe accuracy of Network 1 for Figure 1 is: ',
          accuracy * 100, '%')
    plt.subplot(2, 2, 2)
    plt.scatter(neuron1, neuron2, c=dataSet1[:, 2])
    plotLineY([-1, 1], y_range)  # n1-n2-1=n
    plt.subplot(2, 2, 3)
    plt.scatter(dataSet1[:, 0], dataSet1[:, 1], c=dataSet1[:, 2])
    x_range = [-0.5, 2.5]
    plotLineY([0.5, 0], x_range)  # y=0.5
    plotLineY([1.5, 0], x_range)  # y=-0.5
    neuron1 = dataSet1test@[0, 1]-0.5  # y-0.5=n1
    neuron2 = dataSet1test@[0, 1]-1.5  # y-1.5=n2
    # Activation function if v>0 v=1 if v<=0 v=-1
    for i in range(neuron1.shape[0]):
        if neuron1[i] > 0:
            neuron1[i] = 1
        else:
            neuron1[i] = -1
    for i in range(neuron2.shape[0]):
        if neuron2[i] > 0:
            neuron2[i] = 1
        else:
            neuron2[i] = -1
    neuron = 1 * ((neuron1 - neuron2 - 1) > 0)
    # We exchanged label so the sum is 1
    accuracy = accuracy_score((dataSet1[:, 2]+neuron), np.ones((neuron.shape[0], 1)))
    print('\nThe accuracy of Network 2 for Figure 1 is: ',
          accuracy * 100, '%')
    plt.subplot(2, 2, 4)
    plt.scatter(neuron1, neuron2, c=dataSet1[:, 2])
    plotLineY([-1, 1], y_range)  # n1-n2-1=n


def plotfigure2():
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.scatter(dataSet2[:, 0], dataSet2[:, 1], c=dataSet2[:, 2])
    y_range = [-0.6,1.5]
    plotLineX([0.5, 0], y_range)   # x=0.5
    plotLineX([-0.5, 0], y_range)  # x=-0.5
    neuron1 = dataSet2test@[1, 0]+0.5  # x+0.5=n1
    neuron2 = dataSet2test@[1, 0]-0.5  # x-0.5=n2
    # Activation function if v>0 v=1 if v<=0 v=-1
    for i in range(neuron1.shape[0]):
        if neuron1[i] > 0:
            neuron1[i] = 1
        else:
            neuron1[i] = -1
    for i in range(neuron2.shape[0]):
        if neuron2[i] > 0:
            neuron2[i] = 1
        else:
            neuron2[i] = -1
    plt.subplot(2, 2, 2)
    neuron = neuron1+neuron2+1  # n1+n2+1=n
    # Activation function if v>=0 v=v if v<0 v=0
    for i in range(neuron.shape[0]):
        if neuron[i] < 0:
            neuron[i] = 0
    plt.scatter(neuron, np.zeros((neuron.shape[0], 1)), c=dataSet2[:, 2])
    for i in range(neuron.shape[0]):
        if neuron[i] == 3:
            neuron[i] = 2
    accuracy = accuracy_score(dataSet2[:, 2], neuron)
    print('\nThe accuracy of Network 1 for Figure 2 is: ',
          accuracy * 100, '%')
    plt.subplot(2, 2, 3)
    plt.scatter(dataSet2[:, 0], dataSet2[:, 1], c=dataSet2[:, 2])
    x_range = [-1.5, 1.5]
    plotLineY([0, -2], x_range)  # 2x+y=0
    plotLineY([0, 2], x_range)   # -2x+y=0
    neuron1 = dataSet2test@[2, 1]  # n1=2x+y
    neuron2 = dataSet2test@[-2, 1]  # n2=-2x+y
    # Activation function if v>0 v=1 if v<=0 v=-1
    for i in range(neuron1.shape[0]):
        if neuron1[i] > 0:
            neuron1[i] = 1
        else:
            neuron1[i] = -1
    for i in range(neuron2.shape[0]):
        if neuron2[i] > 0:
            neuron2[i] = 1
        else:
            neuron2[i] = -1
    neuron = 2*neuron1+neuron2  # n =2n1+n2
    # Activation function if v>=0 v=v if v<0 v=0
    for i in range(neuron.shape[0]):
        if neuron[i] < 0:
            neuron[i] = 0
    plt.subplot(2, 2, 4)
    plt.scatter(neuron, np.zeros((neuron.shape[0], 1)), c=dataSet2[:, 2])
    for i in range(neuron.shape[0]):
        if neuron[i] == 3:
            neuron[i] = 1
        elif neuron[i] == 1:
            neuron[i] = 2
    accuracy = accuracy_score(dataSet2[:, 2], neuron)
    print('\nThe accuracy of Network 2 for Figure 2 is: ',
          accuracy * 100, '%')


def plotfigure31():
    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.scatter(dataSet3[:, 0], dataSet3[:, 1], c=dataSet3[:, 2])
    plotLineY([-3, 1.5], [0.5, 4.5])   # 3x-2y-6=0
    plotLineY([-0.8, 1.5], [0.5, 4.5])   # 3x-2y-1.6=0
    plotLineY([1.5, 1.5], [0.5, 4.5])  # 3x-2y+3=0
    # Activation function if v>0 v=1 if v<=0 v=0
    neuron1 = 1*((dataSet3test@[3, -2]+3) > 0)   # n1 = 3x-2y+3
    neuron2 = 1*((dataSet3test@[3, -2]-1.6) > 0)   # n2 = 3x-2y-1.6
    neuron3 = 1*((dataSet3test@[3, -2]-6) > 0)   # n3 = 3x-2y-6
    plt.subplot(2, 1, 2)
    # Activation function if v>0 v=1 if v<=0 v=0
    neuron = 1*(neuron1-1.5*neuron2+neuron3 > 0)  # n=n1-1.5*n2+n3
    plt.scatter(neuron, np.zeros((neuron.shape[0], 1)), c=dataSet3[:, 2])
    # plot 3D result
    fig5 = plt.figure(5)
    ax = Axes3D(fig5)
    X = np.arange(-0.25, 1.25, 0.25)
    Y = np.arange(-0.25, 1.25, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = 1.5*Y-X  # n1-1.5*n2+n3=0
    ax.plot_surface(X, Y, Z)
    ax.scatter(neuron1, neuron2, neuron3, c=dataSet3[:, 2])
    ax.view_init(20, 65)
    accuracy = accuracy_score(dataSet3[:, 2], neuron)
    print('\nThe accuracy of Network 1 for Figure 3 is: ',
          accuracy * 100, '%')


def plotfigure32():
    plt.figure(4)
    plt.subplot(2, 1, 1)
    plt.scatter(dataSet3[:, 0], dataSet3[:, 1], c=dataSet3[:, 2])
    y_range = [-1, 6]
    plotLineX([2.7, 0], y_range)  # x=2.7
    plotLineX([1.5, 0], y_range)  # x=1.5
    plotLineX([3.5, 0], y_range)  # x=3.5
    plt.subplot(2, 1, 2)
    # Activation function if v>0 v=1 if v<=0 v=0
    neuron1 = 1*((dataSet3test@[1, 0]-1.5) > 0)  # n1=x-1.5
    neuron2 = 1*((dataSet3test@[1, 0]-2.7) > 0)  # n2=x-2.7
    neuron3 = 1*((dataSet3test@[1, 0]-3.5) > 0)  # n3=x-3.5
    # Activation function if v>0 v=1 if v<=0 v=0
    neuron = 1*(neuron1-1.5*neuron2+neuron3 > 0)  # n=n1-1.5*n2+n3
    plt.scatter(neuron, np.zeros((neuron.shape[0], 1)), c=dataSet3[:, 2])
    fig6 = plt.figure(6)
    ax = Axes3D(fig6)
    X = np.arange(-0.25, 1.25, 0.25)
    Y = np.arange(-0.25, 1.25, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = 1.5*Y-X  # n1-1.5*n2+n3=0
    ax.plot_surface(X, Y, Z)
    ax.scatter(neuron1, neuron2, neuron3, c=dataSet3[:, 2])
    ax.view_init(20, 65)
    for i in range(neuron.shape[0]):
        if neuron[i] == 3:
            neuron[i] = 2
    accuracy = accuracy_score(dataSet3[:, 2], neuron)
    print('\nThe accuracy of Network 2 for Figure 3 is: ',
          accuracy * 100, '%')


plotfigure1()
plotfigure2()
plotfigure31()
plotfigure32()
plt.show()