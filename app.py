import tensorflow as tf
from keras.models import Sequential
import pandas as pd
import numpy as np

# loading the data
dataset=pd.load_csv("pima-indians-diabetes.csv")

print(dataset.head())

