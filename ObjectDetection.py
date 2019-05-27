# Importing the required libraries and frameworks

import tensorflow as tf
import keras
from keras import backend as K
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)


# Keras.dataset have CIFAR-10 dataset,we will use the dataset from keras and divide them into training set and testing set

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Let’s print the shape of training and testing datasets

print("Train samples:", x_train.shape, y_train.shape)
print("Test samples:", x_test.shape, y_test.shape)