import data_importer
import descriptive_stats as ds
import linear_regression as lr
import financial_data as fd
import LSTM_prediction as lstm
import sentiment_analysis as sa
import google_news as gn

# Import stock data and get user start and end date
stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()
# Show stock data
ds.show_stock_price(stock_ticker)
# Basic descriptive statistics
ds.describe_stock_data(stock_data, stock_ticker)

# Show news headlines
stock_news, headline1, headline2, headline3, headline4, headline5 = gn.get_headlines(stock_ticker)

# Show financial data
fd.scrape_analyst_predictions(stock_ticker)
fd.scrape_financial_kpi(stock_ticker)

# Plot simple trendline
ds.plot_trendline(stock_data, stock_ticker, start_date, end_date)
# Plot moving averages
ds.plot_simple_ma(stock_data, stock_ticker, start_date, end_date)
# Plot weighted moving average
ds.plot_weighted_ma(stock_data, stock_ticker, start_date, end_date)
# Calculate autocorrelation
ds.calculate_autocorrelation(stock_data)
# Plot MACD curve
ds.plot_macd(stock_data, stock_ticker, start_date, end_date)

# Prepare linear regression data
lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(stock_data)
# Do linear regression
lr_line, lr_rsquared = lr.linear_regression(stock_data, stock_ticker, lr_target_date, lr_X, lr_Y)
# Evaluate linear regression
lr.linear_regression_evaluation(lr_Y, lr_line, lr_rsquared)

# Create LSTM Prediction
lstm.lstm_prediction(stock_data, stock_ticker)

# Create RF Prediction
countvector, randomclassifier = sa.train_rf_model()
sa.rf_predict(stock_news, countvector, randomclassifier)
