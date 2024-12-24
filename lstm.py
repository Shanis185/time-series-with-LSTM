import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)

dataframe=pd.read_csv("airline.csv",usecols=[1],engine="python")
dataset=dataframe.values
dataset=dataset.astype('float32')

scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

def create_dataset(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back),0]
        datax.append(a)
        datay.append(dataset[i+look_back,0])
    return numpy.array(dataX),numpy.array(dataY)