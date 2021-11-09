"""
Python Assignment: stock_data_importer.py
This program imports stock data based on user input and saves it as a Pandas Dataframe.
"""

# Import relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data

# Get stock data from Yahoo Finance based on user input
def get_yahoo_data():
    """
    This function gets Yahoo Finance Data based on a stock ticker and a given 
    time-frame and then saves that data as a Pandas Dataframe.
    """
    
    # Ask for user input
    while True:
        try:
            stock_ticker = str(input("Please enter the stock ticker of the stock you want to analyze: "))
            start_date = str(input("Please enter a start date for stock analysis (YYYY-DD-MM): "))
            end_date = str(input("Please enter an end date for stock analysis (YYYY-MM-DD): "))
            # Create dataframe with DataReader module
            stock_data = data.DataReader(stock_ticker, "yahoo", start_date, end_date)
            break
        
        except:
            print("Invalid format for either stock ticker or dates - please try again and ensure correct format.")
    
    
    print(f"Successfully imported stock data for ticker {stock_ticker} from {start_date} to {end_date}.")
    
    return stock_data, stock_ticker, start_date, end_date

def main():
    stock_data, stock_ticker, start_date, end_date = get_yahoo_data()
    
if __name__ == "__main__":
    main()