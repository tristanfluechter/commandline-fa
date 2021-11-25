import modules.data_importer as data_importer
import modules.descriptive_stats as ds
import modules.linear_regression as lr
import modules.financial_data as fd
import modules.LSTM_prediction as lstm
import modules.sentiment_analysis as sa
import modules.google_news as gn
import modules.facebook_prophet as pf

# Import stock data and get user start and end date
stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()

# Plot stock data
ds.plot_stockdata(stock_data)
# Show stock data
ds.show_stock_price(stock_ticker)
# Basic descriptive statistics
ds.describe_stock_data(stock_data, stock_ticker)

# Show news headlines
stock_news = gn.get_headlines(stock_ticker)
gn.print_headlines(stock_news, stock_ticker)

# Show financial data
fd.scrape_analyst_predictions(stock_ticker)
fd.scrape_financial_kpi(stock_ticker)

# Plot simple trendline
ds.plot_trendline(stock_data)
# Plot moving averages
ds.plot_simple_ma(stock_data)
# Plot weighted moving average
ds.plot_weighted_ma(stock_data)
# Plot MACD curve
ds.plot_macd(stock_data, stock_ticker)

# Prepare Regression Data
lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(stock_data)
# Do regression
lr_line, lr_rsquared, reg_pred = lr.linear_regression(lr_target_date, lr_X, lr_Y)
# Evaluate regression
lr.linear_regression_evaluation(lr_Y, lr_line, lr_rsquared)

# Create LSTM dataset
look_back, date_train, date_test, close_train, close_test, train_generator, test_generator, close_data_noarray, close_data = lstm.lstm_prepare_data(stock_data, stock_ticker)
# Train LSTM model
model, prediction, close_train, close_test = lstm.lstm_train(look_back, train_generator, test_generator, close_test, close_train)
# Visualize Model
lstm.lstm_visualize(date_test, date_train, close_test, close_train, prediction, stock_ticker)
# Make Prediction
lstm_pred = lstm.lstm_make_prediction(model, look_back, stock_data, close_data, close_data_noarray, stock_ticker)
# Evaluate Prediction
lstm.lstm_evaluation(prediction, close_train)

# Make randomforest prediction
rf_pred = sa.rf_pred(stock_news)

# Make Prophet Prediction
# Prepare Dataset
prophet_data_train = pf.prophet_forecast(stock_data)
# Create Forecast
m, forecast, prophet_pred = pf.prophet_forecast(prophet_data_train)
# Visualize Forecast
pf.prophet_visualize_forecast(m, forecast)
# Visualize components
pf.prophet_visualize_components(m, forecast)
