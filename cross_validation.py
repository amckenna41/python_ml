import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#import iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#initialise column names
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
