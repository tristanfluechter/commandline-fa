"""
This module uses the stock dataframe to create and train a
LSTM model to predict stock prices at a certain date.

****************************************************************
Credit for the overall usage of an LSTM model for predicting stock prices: https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
All customizing and re-writing as functional code has been done by the authors of this assignment.
****************************************************************
"""

# Import relevant libraries
from matplotlib.pyplot import close
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
from data_importer import get_yahoo_data


def lstm_prediction(stockdata, stockticker):
    """
    This program prepares the data for a LSTM-model by getting a user input on the train-test-split.
    To avoid overfitting, data should always be split into training and testing data. After training a model with one LSTM layer,
    the program visualizes the given predictions. Those predictions are then used to give an estimate of where the stock could be in the next n days.
    """
    while True:
        try:
            # To get a sensible train test split, we restrict the amount of freedom our user gets.
            split_percent = round(float(input("Please enter your desired train-test-split as a positive float between 0.6 and 0.8: ")), 2)
            
            # If split is correct, we break out of the loop
            if split_percent >= 0.6 and split_percent <= 0.8:
                print(f"Successfully set test-train split at {split_percent} test and {round(1-split_percent,2)} train.")
                break
        # User has to try again if the number is not a positive float.    
        except: 
            print("Please enter a positive float number between 0.6 and 0.8.")
    
    # Convert dataframe index (which is a datetime index) to a column
    stockdata['Date'] = stockdata.index

    # Create a closing price list (will be used later!)
    close_data_noarray = stockdata['Close'].values
    
    # Create a closing price array
    close_data = close_data_noarray.reshape((-1,1))
    
    # Split the data into train and test data
    split = int(split_percent*len(close_data))
    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = stockdata['Date'][:split]
    date_test = stockdata['Date'][split:]
    
    # Consider past 15 days
    look_back = 15

    # Generate a dataset to train the LSTM. Each datapoint has the shape of ([past 15 values], [current value]).
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
    
    # Create the LSTM model
    model = Sequential()
    # Add first (and only) layer to the model. 15 nodes, ReLu activation function 
    model.add(LSTM(units = 15, activation='relu', input_shape=(look_back,1)))
    # Create densely connected NN layer
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    # How many epochs do we train the model?
    num_epochs = 25
    # Fit model with data, don't allow status updates
    model.fit(train_generator, epochs=num_epochs, verbose=0)
    # Create prediction values
    prediction = model.predict(test_generator)

    # Reshape train, test and prediction data to plot them
    close_train = close_train.reshape((-1)) 
    close_test = close_test.reshape((-1)) 
    prediction = prediction.reshape((-1))
    
    # Create plotly line 1: Training data
    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    # Create plotly line 2: Testing data
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    # Create plotly line 3: Prediction Values
    trace3 = go.Scatter(
        x = date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = f"{stockticker} Stock",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    
    # Show Figure
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()
    
    # For how many days do we predict?
    # Set to 15 because the LSTM predictions autocorrelate with previous predictions.
    # This way, we avoid overly dramatic movements in either direction.
    num_prediction = 15
    
    prediction_list = close_data[-look_back:] # input scaled data
    
    for number in range(num_prediction):
        
        x = prediction_list[-look_back:] # create list with last look_back values
        x = x.reshape((1, look_back, 1)) # reshape last values so it fits the data
        out = model.predict(x)[0][0] # create a prediction based on those values
        prediction_list = np.append(prediction_list, out) # append this prediction to the output
        
    prediction_list = prediction_list[look_back-1:]
    # Get last date of dataframe
    last_date = stockdata['Date'].values[-1]
    # Create prediction dates time series
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()

    # Plot original values and dates together with predicted values and dates
    trace_original = go.Scatter(
        x = stockdata['Date'][-50:],
        y = close_data_noarray[-50:],
        mode = 'lines',
        name = 'Original Price Curve'
    )

    trace_pred = go.Scatter(
        x = prediction_dates,
        y = prediction_list,
        mode = 'lines',
        name = 'Prediction'
    )
    layout2 = go.Layout(
        title = f"{stockticker} Stock Prediction",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    # Show Figure
    fig2 = go.Figure(data=[trace_original, trace_pred], layout=layout2)
    fig2.show()   

def main():
    lstm_prediction()
  
if __name__ == "__main__":
    main()