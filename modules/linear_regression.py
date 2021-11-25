"""
This module provides the user with a basic linear regression to make a
prediction for a given date.
"""
    
# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
import statsmodels.api as sm
import datetime as dt
import plotly.graph_objects as go
from data_importer import get_yahoo_data


def linear_regression_dataprep(stockdata):
    """
    This program performs a linear regression based on the obtained stock data,
    a user-specified timeframe and a desired prediction date.
    It outputs a graph that includes all stated information.
    """
    
    # Get length of dataframe (ie number of days we can consider for linear regression)
    stockdata_days = len(stockdata)
    
     # Define the last date !
    # Ask user how many days before current date to use for linear regression
    while True:
        try:
            lr_days = int(input("How many past days do you want to consider for the linear regression? (integer) "))
            
            # Check if number of days is between end and start date
            if lr_days < stockdata_days and lr_days >= 0:
                break
            else:
                print(f"Desired {lr_days} days to consider cannot exceed the data time period ({stockdata_days} days).")
                
        except:
            print("Invalid input. Please enter your desired days as a positive integer. Desired days cannot exceed the data time period.")
    
    # Create dataset with the last lr_days days
    lr_dataframe = stockdata.tail(lr_days)
    
    # Ask user for a target date for the regression
    while True:
        try:
            # Get user input for date
            lr_target_date = str(input("Please enter your target date you want to predict the price for (YYYY-MM-DD): "))
        
            # Check if target date earlier than the dataframe
            if dt.datetime.strptime(lr_target_date, '%Y-%m-%d').date() < lr_dataframe.index[-1]:
                print(f"The entered date ({lr_target_date} must in the future (after {lr_dataframe.index[-1]}).")
        
            else:
                break
        
        except:
            # If incorrect input, try again!
            print("Target date needs to be entered in format YYYY-MM-DD. Please re-enter target date.")
        
    # Change datetime to ordinal data for linear regression and define X & Y
    lr_X = np.asarray(lr_dataframe.index.map(dt.datetime.toordinal))
    lr_Y = np.asarray(lr_dataframe.Close)
 
    # Reshape Data into Numpy Array
    lr_X = lr_X.reshape(-1,1)
    lr_Y = lr_Y.reshape(-1,1)
    
    return lr_target_date, lr_X, lr_Y

def linear_regression(targetdate, lr_X, lr_Y):
    """
    This program creates a linear regression with the preprocessed data
    to make a prediction based on a user-input target date.
    
    Returns: lr_line, lr_squared
    """
        
    # Create statsmodels LR object and add lr_X
    x = sm.add_constant(lr_X)
        
    # Predict Results
    lr_results = sm.OLS(lr_Y,x).fit() # variables do not need to be scaled for OLS regression.
        
    # Give summary of linear regression
    lr_results.summary()
        
    # Assign linear regression curve y-intercept and slope based on summary table
    lr_slope = lr_results.summary2().tables[1]["Coef."][1] # to get slope coefficient
    lr_y_intercept = lr_results.summary2().tables[1]["Coef."][0] # to get intercept
    lr_rsquared = lr_results.summary2().tables[0][3][0]
        
    # Convert user input date to ordinal value
    lr_target_date = targetdate.toordinal()
        
    # Append target date to lr_X and lr_Y
    lr_X = np.append(lr_X, lr_target_date)
    lr_Y = np.append(lr_Y, (lr_target_date * lr_slope + lr_y_intercept))
        
    # Create linear regression dataset to plot later
    lr_line = lr_X * lr_slope + lr_y_intercept
        
    # Retransform dates to datetime to create useful x-axis
    # Create shape var to find out current state of lr_X
    x_shape = lr_X.shape
        
    # Create list object from np_array (lr_X)
    lr_X_list = lr_X.reshape(1, x_shape[0])[0].tolist()
        
    # Create list of dates with iteration
    dates = []

    for number in lr_X_list:
        created_date = dt.date.fromordinal(number)
        dates.append(created_date)
    
    # Create plotly object
    fig = go.Figure()
    
    # Plot closing prices
    fig.add_trace(go.Scatter(x=dates, y=lr_Y, mode = "markers", name = "Closing Price"))
    
    # Plot Regression Line for existing values
    fig.add_trace(go.Scatter(x=dates[:-1], y=lr_line[:-1], name="Regression Line",line=dict(color="red"), mode = "lines"))
    # Plot Regression Line prediction
    fig.add_trace(go.Scatter(x=dates[-2:], y=lr_line[-2:], name=f"Regression Line Prediction", line=dict(color="red", dash="dash")))
    
    # Format layout (Show legend in top left corner,
    # remove title margins, remove backgroud colour,
    # label the axes
    fig.layout.update(showlegend = True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), 
                      margin=go.layout.Margin(l=60, r=0, b=0, t=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title="Date"),yaxis=dict(title="Closing Price in USD"))
    
    # Format axes
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    
    # Add annotation to prediction date
    fig.add_annotation(x=dates[-1], y=(lr_line[-1]+1),
            text="Predicted Date")
    
    # Show plot
    fig.show()
    
    # Create final values
    reg_pred = int(lr_line[-1])
    
    return lr_line, lr_rsquared, reg_pred

def linear_regression_evaluation(lr_Y, lr_line, lr_rsquared):
    """
    Evaluates the linear regression accuracy and the level at which it
    can give insights into the stock price movement.
    """
    # Calculate RMSE based on lr_line (regression) and lr_y (actual values)
    root_mean_square_error = np.sqrt(((lr_line - lr_Y) ** 2).mean()).round(2)
    # RSquared was returned in the Statsmodels OLS Regression summary
    r_squared = float(lr_rsquared)
    
    # Print out results
    print(f"Linear Regression RMSE: {root_mean_square_error}")
    print(f"Linear Regression R-Squared: {r_squared}")
    
    # Evaluate r-squared metric - how much of the movements does the regression explain?
    if r_squared <= 0.4:
        print(f"With an r-squared value of {r_squared}, it is __not sufficient__ to rely on a simple regression to predict stock values.")
    else:
        print(f"With an r-squared value of {r_squared}, the regression seems to __identify a trend in stock prices__. However, we advise to use additional predictive measures.")