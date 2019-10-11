In this project, I write four *.py files. The main procedure is according to characteristics of HSV space and Gaussian pdf to get threshold value.
Then use KNN to filter again to make the results more accurate. Finally, remove close data and return locations.

project00.py:
It is a entire code for all my project and return results of two full picture. You can run it if you are interested in it.
It takes about 1000 seconds to finish. And predicted locations in 'data_train_prediction.npy' and 'data_test_prediction.npy'
For more details please see description in code.

In fact I split project00 into three parts: train.py, test.py and run.py

train.py:
In this file, firstly I load some neccessary dataset like 'error_pixel.npy' and 'red_car.npy' which are created by myself.
There are five functions.
'RBG2HSV'  converts RGB value to HSV space. Input argument is a RGB value and return HSV value
'trainHSV' trains data from ground_truth and get some threshold value for h, s and v. Here I use threshold value because of 
	   charateristics of HSV space. Return threshold value.
'traindataKNN' initializes KNN parameters, add error_pixel and red_car to train dataset to make up a new train dataset for KNN.
	       This function returns HSV value of train dataset.
'labelKNN' initializes label for KNN. Return laebl dataset
'KNN' utilizes data and labels got from 'traindataKNN' and 'labelKNN', returns KNN's module

test.py:
In this file, I load parameters from train.py and get predicted value.
There are four functions. And 'predict' is a main function. Other three functions are used in 'predict'
'RGB2HSV' same with above
'calculateddistance' calculate distance of each rows in a matrix. Input matrix(m*n) and matrix(m*n) itself and return a matrix(m*m).
		     The i-row j-column of new matrix indicates distance between i-row and j-row in original matrix.
'KNNerrorkiller' is a function I used to filter my result again to delete some error points such as red roofs or something like that.
		 Input dataset of locations of red pixels and your test picture dataset. Return locations which meet my requirements.
'predict' is a main function to predict red cars' location in a given picture. You just input dataset of your picture and it will return 
	  locations of red cars.

run.py:
OK. Welcome to the final part. In this file I load dataset of data_test and choose a small sample. Then just use 'predict' and get result!   


ALL IN ALL. What you should do is get 'error_pixel.npy', 'red_car.npy' and also the three offered *.npy, and train.py, test.py, run.py.
And I can not upload "data_train.npy" and "data_test.npy", please use yours. Put them in same directory and run run.py and get result. 
What's more, 'data_train_prediction.npy' and 'data_test_prediction.npy' contains predicted locations in the two full picture. 
