"""
This module uses a created Pandas Dataframe to conduct descriptive statistics.
"""
# Import relevant libraries
from yahoo_fin import stock_info as si
import pandas as pd
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import data_importer

def show_stock_price(ticker):
    """
    A program that returns the current stock price.
    """
    current_stock_price = si.get_live_price(ticker).round(2)
    print(f"Current stock price: {current_stock_price} USD")
    return current_stock_price


def describe_stock_data(stockdata, ticker):
    """
    A program that describes the stock data, 
    providing basic descriptive statistics.
    """
    
    # Save new dataframe for descriptive statistics
    descriptive_df = stockdata.describe().Close
    
    # Get descriptive variables through indexing
    stock_des_mean = descriptive_df['mean'].round(2)
    stock_des_quart1 = descriptive_df['25%'].round(2)
    stock_des_quart2 = descriptive_df['50%'].round(2)
    stock_des_quart3 = descriptive_df['75%'].round(2)
    stock_des_stddev = descriptive_df['std'].round(2)
    stock_des_range = (stock_des_quart3 - stock_des_quart1).round(2)
    stock_des_var_coefficient = ((stock_des_stddev / stock_des_mean) * 100).round(2)
    
    # Print descriptive statistics for the stock.
    print(f"The mean closing price of stock {ticker} is {stock_des_mean}.")
    print(f"The first quartile of stock {ticker}'s closing price is {stock_des_quart1}.")
    print(f"The second quartile of stock {ticker}'s closing price is {stock_des_quart2}.")
    print(f"The third quartile of stock {ticker}'s closing price is {stock_des_quart3}.")
    print(f"That means the range is equal to {stock_des_range}.")
    print(f"The stock's closing price shows a standard deviation of {stock_des_stddev} and a variation coefficient of {stock_des_var_coefficient}")


def plot_trendline(stockdata, ticker, startdate, enddate):
    """
    A program that plots the ticker data over the given timeframe
    and provides a linear trendline.
    """
    # Create matplotlib plot object        
    fig = plt.figure(figsize=(12,6))
    
    # Plot closing prices
    plt.plot(stockdata.Close, label="Closing Price")
    
    # Convert Date Axis to numerical for trend line
    numeric_dates = mdates.date2num(stockdata.index)
    
    # Fit data to create trend line
    fitted_data = np.polyfit(numeric_dates, stockdata.Close, 1)
    trend_curve = np.poly1d(fitted_data)
    
    # Plot trend line
    plt.plot(numeric_dates, trend_curve(numeric_dates), "y--", label="Trend Line")
    
    # Name axes
    plt.xlabel('Date')
    plt.ylabel('Closing Price in USD')
    
    # Add title
    plt.title(f"Closing price of {ticker} from {startdate} to {enddate} with trend line.")
    
    # Add legend and set location to top left
    plt.legend(loc="upper left")
    
    #Show the graph
    plt.show()


def plot_simple_ma(stockdata, ticker, startdate, enddate):
    """
    A program that plots the ticker data over the given timeframe
    and provides moving averages based on user input.
    """
    
    # Define moving averages to plot
    while True:
        try:    
            ma1_input = int(input("Please state a first moving average to plot (in days): "))
            ma2_input = int(input("Please state a second moving average to plot (in days): "))
            break
        except: 
            print("Invalid input. Please state the desired moving average in days.")
    
    # Create matplotlib plot object        
    fig = plt.figure(figsize=(12,6))
    
    # Plot closing prices
    plt.plot(stockdata.index, stockdata.Close, label="Closing Price")
    
    # Get user input for moving averages
    ma1 = stockdata.Close.rolling(ma1_input).mean()
    ma2 = stockdata.Close.rolling(ma2_input).mean()
    
    # Plot moving averages
    plt.plot(stockdata.index, ma1, "g", label=f"Moving Average: {ma1_input} days.") # Plot first MA in green
    plt.plot(stockdata.index, ma2, "r", label=f"Moving Average: {ma2_input} days.") # Plot second MA in red
    
    # Give description to axes / values
    plt.xlabel('Date')
    plt.ylabel('Closing Price in USD')
    plt.title(f"Closing price of {ticker} from {startdate} to {enddate} with moving averages.")
    plt.legend(loc="upper left")
    
    # Show graph
    plt.show()

def calculate_autocorrelation(stockdata):
    """
    This program calculates the amount of autocorrelation of a given stock
    to give additional insights for short-term traders.
    """
    autocorrelation_plot(stockdata.Close)
    plt.title("Autocorrelation Values for Time Series")
    plt.show()

def plot_weighted_ma(stockdata, ticker, startdate, enddate):
    """
    A program that plots the ticker data over the given timeframe
    and provides a n-day weighted moving average based on user input.
    """
    # Get number of days for the custom weighted average.
    while True:
    
        try:
            wa_days = int(input("How many days for your custom weighted average? (up to 20 days) "))

            if wa_days <= 20 and wa_days > 0: # ensure the number of days make sense
                
                print(f"Selected number of days: {wa_days}")
                
                while True:      
                    # Define empty list and weight index for user input. Resets if weights don't add up to 1.
                    weights_for_ma = []
                    weight_index = 1
            
                    # Iterate through the user-stated days
                    for number in range(wa_days):
                        
                        # Check for each user-input number that it is a float.
                        while True:
                            
                            try:
                                # If wrong, user needs to re-enter the float.
                                input_weight = float(input(f"Please enter weight #{weight_index}: "))
                                    
                                if input_weight >= 0 and input_weight <= 1:
                                    # Append weights list
                                    weights_for_ma.append(input_weight)
                                    # Increase index for print statement
                                    weight_index += 1
                                    print(f"Current total amount of weights: {round(sum(weights_for_ma), 2)}")
                                    break # breaks out of innermost while loop and user can input the next number.
                                        
                                else:
                                    print("Invalid input. Entered float must be positive and between 0 and 1.")
                                    
                            except:
                                print("Invalid input. Please input your weight as a float number.")

                    if int(sum(weights_for_ma)) == 1:
                        print("Weights have been set.")
                        break # breaks out of weights list loop
                    
                    else:
                        print("Sum of weights must be equal to 1. Please re-enter your weights.") # try again
                
                break # breaks out of initial top tier loop
                
            else:
                print("Invalid input. Please input less or equal to 20 and larger than 1.") # Try again
    
        except:
            print("Invalid input. Please input the number of days as an integer less or equal to 20 and larger than 1.")
    
    # Transform weights into numpy array
    weights_for_ma = np.array(weights_for_ma)
    ma_weighted = stockdata.Close.rolling(wa_days).apply(lambda x: np.sum(weights_for_ma * x))
    
    # Create matplotlib plot object        
    fig = plt.figure(figsize=(12,6))
    
    # Plot closing prices
    plt.plot(stockdata.index, stockdata.Close, label="Closing Price")
    
    # Plot weighted moving averages
    plt.plot(ma_weighted, "y", label=f"{wa_days}-Day Weighted Moving Average")
    
    # Give description to axes / values
    plt.xlabel('Date')
    plt.ylabel('Closing Price in USD')
    plt.title(f"Closing price of {ticker} from {startdate} to {enddate} with moving averages.")
    plt.legend(loc="upper left")
    
    # Show graph
    plt.show()
    

def plot_macd(stockdata, ticker, startdate, enddate):
    """
    Code credit: https://www.alpharithms.com/calculate-macd-python-272222/
    This program plots the price chart combined with a moving average convergence / divergence.
    """
    
    # Check if timeframe long enough for MACD analysis
    while True:
        try: 
            # Calculate MACD values and append them to stockdata dataframe
            stockdata.ta.macd(close='Close', fast=12, slow=26, append=True)
            break
        except:
            print("Entered timeframe not long enough to do MACD study.")
    
    # TO DO: BREAK HERE AND GO TO NEXT STEP IF NOT POSSIBLE!
    
    # Generate plotly plot object
    fig = make_subplots(rows=2, cols=1, subplot_titles=[f"Candlechart: Ticker {ticker} over time.", "MACD"],)
    
    # Append closing price line to first graph
    fig.append_trace(go.Scatter(x=stockdata.index, y=stockdata['Open'],line=dict(color='black', width=1),
        name='Open', legendgroup='1',), row=1, col=1)

    # Append candlesticks for first graph
    fig.append_trace(go.Candlestick(x=stockdata.index, open=stockdata['Open'], high=stockdata['High'], low=stockdata['Low'],
        close=stockdata['Close'], increasing_line_color='green', decreasing_line_color='red', showlegend=False), 
        row=1, col=1)
    
    # Append Fast Signal (%k) line to second graph
    fig.append_trace(go.Scatter(
        x=stockdata.index,
        y=stockdata['MACD_12_26_9'],
        line=dict(color='Blue', width=2),
        name='MACD',
        # showlegend=False,
        legendgroup='2',), row=2, col=1)
    
    # Append Slow signal (%d) line to second graph
    fig.append_trace(go.Scatter(
        x=stockdata.index,
        y=stockdata['MACDs_12_26_9'],
        line=dict(color='Orange', width=2),
        # showlegend=False,
        legendgroup='2',
        name='Signal'), row=2, col=1)
    
    # Colorize the data to emphasize difference between fast and slow signal
    colors = np.where(stockdata['MACDh_12_26_9'] < 0, 'red', 'green')
    
    # Append colorized histogram to second graph indicating difference between fast and slow signal
    fig.append_trace(go.Bar(x=stockdata.index, y=stockdata['MACDh_12_26_9'], name='Histogram', marker_color=colors), row=2, col=1)
    
    # Define layout for graphs: Font size and rangeslider.
    layout = go.Layout(font_size=14, xaxis=dict(rangeslider=dict(visible=False)))
    
    # Update options and show plot
    fig.update_layout(layout)
    fig.show()
    
    
def main():
    stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()
    show_stock_price(stock_ticker)
    describe_stock_data(stock_data, stock_ticker)
    plot_trendline(stock_data, stock_ticker, start_date, end_date)
    calculate_autocorrelation(stock_data)
    plot_simple_ma(stock_data, stock_ticker, start_date, end_date)
    plot_weighted_ma(stock_data, stock_ticker, start_date, end_date)
    plot_macd(stock_data, stock_ticker, start_date, end_date)

if __name__ == '__main__':
    main()