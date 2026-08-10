import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = np.array(np.loadtxt("data/mnist_train.csv", delimiter=","))
print(data.shape)