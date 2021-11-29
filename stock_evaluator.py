"""
This is a stock evaluator and predictor that can be used to get information about
a certain stock over a user-defined timeframe. It uses descriptive statistics and 
web-crawling to give first insights into the stock - based on these statistics, the user can then
implement various predictive models - linear regression, ARIMA, LSTM and a sentiment analysis
to derive a decision on whether or not to buy the stock.

Created by: Tristan Fluechter, Odhrán McDonnell, Anirudh Bhatia, Kunal Gupta
"""

# Import relevant libraries
import os
import modules.data_importer as data_importer
import modules.descriptive_stats as ds
import modules.linear_regression as lr
import modules.financial_data as fd
import modules.LSTM_prediction as lstm
import modules.sentiment_analysis as sa
import modules.google_news as gn
import modules.facebook_prophet as pf
import warnings
warnings.filterwarnings('ignore')

# Define App Startup Screen
def startup():
    """
    Program to import a stock dataframe through an API call from Yahoo Finance. Available stock data are all tickers
    that are listed on Yahoo Finance. Error handling in data_importer module ensures valid timeframe.
    """
    
    # CLEAR WINDOW
    os.system("clear")
    print("Welcome to our stock predictor! Provide a stock ticker and a timeframe to perform thorough analysis!")
    stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()
    
    return stock_data, stock_ticker, start_date, end_date

def pred_menu_text():
    print("\nChoose service you want to use : ")
    print("""
        1 : Linear Regression Prediction
        2 : LSTM Prediction
        3 : Random Forest Sentiment Analysis
        4 : Facebook Prophet Prediction
        5 : Return to main menu
        6 : Exit App
        """)

def desc_menu_text():
    print("\nChoose service you want to use : ")
    print("""
        1 : Basic Descriptive Stats
        2 : Trendlines
        3 : Moving Average
        4 : Weighted Moving Average
        5 : MACD Curve
        6 : Return to main menu
        7 : Exit App
        """)

# Define Descriptive Sub-Menu
def descriptive_stats_menu(stock_data, stock_ticker):
    """
    Sub-menu for descriptive stats with the option to either 
    go back to main menu or exit the application
    """
    
    # While-Loop to keep displaying if wrong input_weight
    choice = 0
    
    # While not quit
    while choice != "7":
        # Display menu text
        desc_menu_text() 
        # Get user choice
        choice = input("\nEnter your choice : ")
        
        # Do if choice
        if choice == '1':
            ds.describe_stock_data(stock_data, stock_ticker)
            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice    
        elif choice == '2' : 
            print("\n")
            # Plot simple trendline
            ds.plot_trendline(stock_data)
            print("Successful output of trendline in browser window.")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice    
        elif choice == '3' :
            print("\n")
            # Plot moving averages
            ds.plot_simple_ma(stock_data)
            print("Successful output of MA in browser window.")
            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice    
        elif choice == '4' :
            print("\n")
            # Plot weighted moving average
            ds.plot_weighted_ma(stock_data)
            print("Successful output of weighted MA in browser window.")
            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice    
        elif choice == '5' :
            print("\n")
            # Plot MACD curve
            ds.plot_macd(stock_data, stock_ticker)
            print("Successful output of MACD in browser window.")
            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice    
        elif choice == '6' :
            # Go back to main menu
            break
            
        # Do if choice    
        elif choice == '7' :
            # Exit application
            exit()
             
        else:
            print("Wrong input format. Please re-enter.")
            # User makes new input
            continue

# Define Predictive Sub-Menu
def predictive_stats_menu(stock_data, stock_ticker):
    """
    Sub-menu for predictive stats with the option to either 
    go back to main menu or exit the application
    """
    
    # While-Loop to keep displaying if wrong input
    choice = 0
    
    # While not quit
    while choice != "7":
        
        # Show menu text
        pred_menu_text()
        
        # Get user choice
        choice = input("\nEnter your choice : ")
        
        # Do if choice
        if choice == "1":
            
            try:
                print("\n")
                # Prepare Regression Data
                lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(stock_data)
                print("Preparing data for linear regression...")
                # Do regression
                lr_line, lr_rsquared, reg_pred = lr.linear_regression(lr_target_date, lr_X, lr_Y)
                print("Successful output of regression prediction data in browser window.")
                # Evaluate regression
                lr.linear_regression_evaluation(lr_Y, lr_line, lr_rsquared)
            
            except:
                print("Timeframe not suitable for linear regression prediction. Please enter timeframe of minimum 100 days length.")
            
            print("\n")
            # Continue analysis or quit?
            continue_choice()
        
        # Do if choice
        elif choice == "2":
            
            try:
                print("\n")
                # Create LSTM dataset
                look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data = lstm.lstm_prepare_data(stock_data)
                print("Training LSTM model...")
                # Train LSTM model
                model, prediction, close_train, close_test = lstm.lstm_train(look_back, train_generator, test_generator, close_test, close_train)
                # Visualize Model
                lstm.lstm_visualize(date_test, date_train, close_test, close_train, prediction)
                print("Successful output of LSTM training in browser window.")
                # Make Prediction
                lstm.lstm_make_prediction(model, look_back, stock_data, close_data, close_data_noarray, stock_ticker)
                print("Successful output of LSTM prediction in browser window.")
                # Evaluate Prediction
                lstm.lstm_evaluation(prediction, close_train)
            except:
                print("Timeframe not suitable for LSTM prediction. Please enter timeframe of minimum 100 days length.")

            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice   
        elif choice == "3":
            print("\n")
            print("Gathering news headlines and testing them with Random Forest model...")
            stock_news = gn.get_headlines(stock_ticker)
            sa.rf_predict(stock_news)
            
            print("\n")
            # Continue analysis or quit?
            continue_choice()
    
        # Do if choice
        elif choice == "4":
            
            try:
                print("\n")
                # Make Prophet Prediction
                # Prepare Dataset
                prophet_data_train = pf.prophet_dataprep(stock_data)
                print("Training Facebook Prophet model...")
                # Create Forecast
                m, forecast, prophet_pred = pf.prophet_forecast(prophet_data_train)
                # Visualize Forecast
                pf.prophet_visualize_forecast(m, forecast)
                print("Successful output of Facebook Prophet prediction in browser window.")
                # Visualize components
                pf.prophet_visualize_components(m, forecast)
                print("Successful output of Facebook Prophet components analysis in browser window.")
            
            except:
                print("\n")
                print("Timeframe not suitable for Facebook Prophet prediction. Please enter timeframe of minimum 100 days length.")
            
            print("\n")
            # Continue analysis or quit?
            continue_choice()  
    
        # Do if choice
        elif choice == "5":
            # Get back to main menu 
            break
        
        # Do if choice
        elif choice == "6":
            # Exit application
            exit()
        
        else:
            # User has to make new input
            print("\n")
            print("Wrong input format. Please re-enter.")
            continue

def print_menu_text():
    
    print("\nChoose service you want to use : ")
    print("""
        1 : Plot Stock Price
        2 : Show News Headlines
        3 : Show Analyst Predictions
        4 : Show Descriptive Statistics Menu
        5 : Show Predictive Statistics Menu
        6 : Pick New Stock and Timeframe
        7 : Exit App
        """)

def continue_choice():
    """
    Asks user if they want to continue analysis or quit. 
    Clears terminal window afterwards.
    """
    
    cont_choice = ""
    
    while cont_choice not in ["Y", "N"]:
        cont_choice = str(input("Want to explore the stock data further (Y/N): "))
                
        if cont_choice.lower() == "y":
            break
        
        elif cont_choice.lower() == "n":
            quit()
        
        else:
            print("Invalid input. Please enter Y or N.")
            continue
    
    os.system("clear")        

# Define Overall Menu Structure
def main_menu(stock_data, stock_ticker, start_date, end_date):
    """
    Main menu for stock evaluator. Takes in variables from startup().
    Offers to either pursue some basic statistics or move into the descriptive / predictive sub-menus.
    Allows user to re-enter stock data.
    """
    
    # While Loop to keep menu running 
    choice = 0
    
    # While not exit
    while choice != '7':
        
        print_menu_text()
        
        # Get user choice
        choice = input("Enter your choice : ")

        # Do if choice
        if choice == '1':
            # Plot stock data
            print("\nSuccessful output of stock data in browser window.")
            ds.plot_stockdata(stock_data)
            # Show stock price
            ds.show_stock_price(stock_ticker)
            # Continue analysis or quit?
            print("\n")
            continue_choice()
    
        # Do if choice     
        elif choice == '2' :
            print("\n")
            # Show news headlines
            stock_news = gn.get_headlines(stock_ticker)
            gn.print_headlines(stock_news, stock_ticker)
            # Continue analysis or quit?
            print("\n")
            continue_choice()
        
        # Do if choice    
        elif choice == '3' :
            # Show financial data
            fd.scrape_analyst_predictions(stock_ticker)
            fd.scrape_financial_kpi(stock_ticker)
            print("\n")
            # Continue analysis or quit?
            continue_choice()
        
        # Do if choice    
        elif choice == '4' :
            # Show descriptive stats menu
            descriptive_stats_menu(stock_data, stock_ticker)
        
        # Do if choice    
        elif choice == '5' :
            # Show predictive stats menu
            predictive_stats_menu(stock_data, stock_ticker)
        
        # Do if choice
        elif choice == '6' :
            # Back to square 1
            startup()
        
        # Do if choice    
        elif choice == '7' :
            # Exit application
            exit()
        
        else:
            print("Wrong input format. Please re-enter.")
            continue

            
stock_data, stock_ticker, start_date, end_date = startup()
main_menu(stock_data, stock_ticker, start_date, end_date)