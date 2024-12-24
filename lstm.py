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

dataframe = pd.read_csv(r"airline_passenger.csv", usecols=[1], engine="python")
dataset = dataframe.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainx, trainy = create_dataset(train, look_back)
testx, testy = create_dataset(test, look_back)

trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
testx = np.reshape(testx, (testx.shape[0], 1, testx.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainx, trainy, epochs=100, batch_size=1, verbose=2)

trainpredict = model.predict(trainx)
testpredict = model.predict(testx)

trainpredict = scaler.inverse_transform(trainpredict)
trainy = scaler.inverse_transform([trainy])
testpredict = scaler.inverse_transform(testpredict)
testy = scaler.inverse_transform([testy])

trainscore = math.sqrt(mean_squared_error(trainy[0], trainpredict[:, 0]))
testscore = math.sqrt(mean_squared_error(testy[0], testpredict[:, 0]))

trainpredictplot = np.empty_like(dataset)
trainpredictplot[:, :] = np.nan
trainpredictplot[look_back:len(trainpredict) + look_back, :] = trainpredict

testpredictplot = np.empty_like(dataset)
testpredictplot[:, :] = np.nan
testpredictplot[len(trainpredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testpredict

plt.plot(scaler.inverse_transform(dataset), label="Original Data")
plt.plot(trainpredictplot, label="Training Predictions")
plt.plot(testpredictplot, label="Test Predictions")

plt.text(len(trainpredict) + look_back, trainpredict[0], f'Train RMSE: {trainscore:.2f}', color='blue', fontsize=12)
plt.text(len(testpredict) + look_back * 2, testpredict[0], f'Test RMSE: {testscore:.2f}', color='red', fontsize=12)

plt.xlabel('Time')
plt.ylabel('Passenger Count')
plt.title('Airline Passenger Prediction using LSTM')
plt.legend()
plt.show()
