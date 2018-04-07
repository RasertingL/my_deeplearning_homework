import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

#%matplotlib inline
np.random.seed(1)