import unittest
import modules.data_importer as data_importer
import modules.descriptive_stats as ds
import modules.linear_regression as lr
import modules.financial_data as fd
import modules.LSTM_prediction as lstm
import modules.sentiment_analysis as sa
import modules.google_news as gn
import modules.facebook_prophet as pf
import pandas as pd
import datetime
import numpy as np
import os

# Define global variables
stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()
cwd = os.getcwd()

class Test_StockEvaluator(unittest.TestCase):
    
    def test_data_importer(self):
        # Check if input is stock ticker
        self.assertIsInstance(stock_ticker, str)
        # Check if stock_data is a pd.DataFrame
        self.assertIsInstance(stock_data, pd.DataFrame)
        # Check for strings
        self.assertIsInstance(start_date, str)
        self.assertIsInstance(end_date, str)
    
    def test_google_news(self):
        stock_news = gn.get_headlines(stock_ticker)
        # Ensure output is a list
        self.assertIsInstance(stock_news, list)
        # Ensure 25 headlines are imported
        self.assertEqual(len(stock_news), 25)   
    
    def test_financial_data(self):
        median_price = fd.scrape_analyst_predictions(stock_ticker)
        # Check if median price is str
        self.assertIsInstance(median_price, str)
        
        abs_change, percent_change = fd.scrape_financial_kpi(stock_ticker)
        # Check if change is a str
        self.assertIsInstance(abs_change, str)
        # Check if change is a str
        self.assertIsInstance(percent_change, str)
    
    def test_linear_regression(self):
        lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(stock_data)
        # Check if target date is in the future
        self.assertIsInstance(lr_target_date, str)
        
        lr_line, lr_rsquared, reg_pred = lr.linear_regression(lr_target_date, lr_X, lr_Y)
        self.assertIsInstance(lr_rsquared, str)
        self.assertIsInstance(reg_pred, int)
          
    def test_lstm(self):
        look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data = lstm.lstm_prepare_data(stock_data)
        # Ensure correct lookback
        self.assertEqual(look_back, 15)
        # Ensure len of train is higher than test
        self.assertTrue(len(date_train) > len(date_test))
        self.assertTrue(len(close_train) > len(close_test))
        
        model, prediction, close_train, close_test = lstm.lstm_train(look_back, train_generator, test_generator, close_test, close_train)
        lstm_pred = lstm.lstm_make_prediction(model, look_back, stock_data, close_data, close_data_noarray, stock_ticker)
        self.assertIsInstance(lstm_pred, int)
        
    def test_sentiment_analysis(self):
        # Check if filepaths are correct
        self.assertTrue(os.path.isfile(cwd + '/data/vector.pickel'))
        self.assertTrue(os.path.isfile(cwd + '/data/randomforest_sentiment_classifier.sav'))
        
        # Check if desired output
        stock_news = gn.get_headlines(stock_ticker)
        rf_pred = sa.rf_predict(stock_news)
        self.assertIsInstance(rf_pred, str)
    
    def test_prophet(self):
        prophet_data_train = pf.prophet_dataprep(stock_data)
        m, forecast, prophet_pred = pf.prophet_forecast(prophet_data_train)
        # Check if desired output
        self.assertIsInstance(prophet_pred, int)

if __name__ == '__main__':
    unittest.main()