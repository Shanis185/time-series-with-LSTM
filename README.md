# Time Series Forecasting with LSTM
#### This project demonstrates how to forecast time series data using Long Short-Term Memory (LSTM) networks. The goal is to predict future values in a time series dataset, specifically airline passenger data, using LSTM, which is a type of recurrent neural network (RNN) that is well-suited for time series forecasting due to its ability to retain information across time steps.

## Project Overview
#### In this project, the goal is to predict the number of airline passengers for future months based on historical data using LSTM. The steps involved include loading and preprocessing the dataset, preparing the data for input to the LSTM model, training the model, and evaluating its performance. The project also visualizes the predictions against the original dataset to assess the accuracy of the model.

## Dataset
#### This project uses the Airline Passenger Dataset, which contains monthly totals of international airline passengers from 1949 to 1960.
#### The dataset has a single feature, representing the number of passengers, and is used to train and test the LSTM model.

## Model Architecture
#### The model is built using Keras, a high-level neural network library running on top of TensorFlow. The architecture is a simple LSTM network:

#### LSTM Layer: A single LSTM layer with 4 units.
#### Dense Layer: A Dense layer with a single neuron to output the predicted value.
#### The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function. The LSTM is trained for 100 epochs using a batch size of 1.

## Steps to Run
### Load the dataset: Read the airline passengers data from a CSV file.
### Preprocess the data:
### Normalize the dataset using MinMaxScaler.
### Split the dataset into training and testing sets (67% training, 33% testing).
### Prepare the data: Convert the data into a supervised learning format suitable for LSTM.
### Reshape the data: The input for the LSTM needs to be reshaped into 3D arrays (samples, time steps, features).
### Build the LSTM model: Create the LSTM model with 4 LSTM units followed by a Dense layer.
### Train the model: Fit the LSTM model on the training data.
### Make predictions: Use the trained model to make predictions on the training and testing data.
### Evaluate the model: Calculate the Root Mean Squared Error (RMSE) for both the training and testing data.
### Visualize the results: Plot the original data along with the training and testing predictions.

## Evaluation
#### The performance of the model is evaluated using Root Mean Squared Error (RMSE), which measures the difference between the predicted and actual values. The lower the RMSE, the better the model is at forecasting.

## Results
#### The results will be displayed in a plot, showing:

#### The original time series data (airline passenger count).
#### The training predictions.
#### The test predictions.
