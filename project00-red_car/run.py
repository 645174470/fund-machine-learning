
""" =======================  Import dependencies ========================== """

import test
import numpy as np
import matplotlib.pyplot as plt
import time

""" =======================  Import DataSet ========================== """

# data_train = np.load("data_train.npy")
data_test = np.load("data_test.npy")

""" =======================  Predict part ========================== """
data_test = data_test[0:3500, :, :]
start = time.clock()  # Timer
testresult = test.predict(data_test)
plt.imshow(data_test)
plt.scatter(testresult[:, 1], testresult[:, 0])
# plt.figure(2)
# plt.imshow(data_train)
# plt.scatter(redcarlocation[:, 1], redcarlocation[:, 0])
# np.save('data_train_small.npy',redcarlocation)
# np.save('data_train_prediction.npy', redcarlocation)
# np.save('data_test_prediction.npy', testresult)
end = time.clock()  # Timer
interval = end - start  # Get run time
print(interval)
