Please make sure you have opencv and pytorch on your computer!
Please download 'cnn.pkl' which is a trained cnn. 

train.py 
Defines some functions and train CNN
picreshape: import data and return normalized data whose size is 60*60
Class CNN: CNN has two convolotion layer.
Do until presupposed Epoch:
    Forward Propagation:
	First Convolution layer
		do 2D-convolotion
		Dropout some neuron  
		Activation function (ReLU)
		Pooling layer (max)
	Second Convolution layer
		do 2D-convolotion
		Dropout some neuron
		Activation function (ReLU)
		Pooling layer (max)
	Fully connected layer
	Output
    Backward Propagation:
	Choose optimizer (Adam)
	Calculate loss between output and true label (CrossEntropyLoss)
	Derivative and make it to zero
	Backward Propagation
Get desired CNN
Then I save it in 'cnn.pkl'

test.py
picreshape, CNN: same with train.py
Use 'cnn.pkl' to predict

I also upload knn.py which is a code for knn



	

