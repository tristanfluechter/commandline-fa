"""
Python Assignment: stock_data_importer.py
This program imports stock data based on user input and saves it as a Pandas Dataframe.
"""

# Import relevant libraries
import pandas_datareader as data
import datetime as datetime

# Get stock data from Yahoo Finance based on user input
def get_yahoo_data():
    """
    This function gets Yahoo Finance Data based on a stock ticker and a given 
    time-frame and then saves that data as a Pandas Dataframe.
    """
    
    # Ask for user input
    while True:
        
        try:
            # Get user inputs for ticker and start / end dates
            stock_ticker = str(input("Please enter the stock ticker of the stock you want to analyze: "))
            start_date = str(input("Please enter a start date for stock analysis (YYYY-MM-DD): "))
            end_date = str(input("Please enter an end date for stock analysis (YYYY-MM-DD): "))
            
            # Create variable to check if target date is in the future
            today = datetime.date.today()
            
            # Convert user inputs to datetime.date object
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            
            # Get days difference between end date and today (should be negative if correct value is input!)
            valid_date_input = (end_dt - today).days
            
            # If end date input is in the future
            if valid_date_input >= 0:
                # Try again to get user input
                print("Invalid input - end date can not lie in the future.")
                continue
            
            else:    
                # Create dataframe with DataReader module
                stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
                break
        
        except:
            print("Invalid format for either stock ticker or dates - please try again and ensure correct format.")
    
    # Calculate timeframe length
    date_difference = (end_dt - start_dt).days
    
    # Report back if timeframea long enough for full functionality
    if date_difference <= 60:
        # Prediction models require more days to work
        print(f"For optimal functionality, please enter a timeframe of more than 60 days (current: {date_difference} days).")
    
    # Tell user successful import has been achieved
    print(f"Successfully imported {date_difference} days of stock data for ticker {stock_ticker} from {start_date} to {end_date}.")
    
    return stock_data, stock_ticker, start_date, end_date