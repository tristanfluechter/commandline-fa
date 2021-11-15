import data_importer
import descriptive_stats as ds
import linear_regression as lr
import financial_data as fd
import google_news as gn

# Import stock data and get user start and end date
stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()
# Show stock data
ds.show_stock_price(stock_ticker)
# Basic descriptive statistics
ds.describe_stock_data(stock_data, stock_ticker)
# Show news headlines
gn.get_headlines(stock_ticker)
# Show financial data
fd.scrape_analyst_predictions(stock_ticker)
fd.scrape_financial_kpi(stock_ticker)
# Plot simple trendline
ds.plot_trendline(stock_data, stock_ticker, start_date, end_date)
# Plot moving averages
ds.plot_simple_ma(stock_data, stock_ticker, start_date, end_date)
# Plot weighted moving average
ds.plot_weighted_ma(stock_data, stock_ticker, start_date, end_date)
# Plot MACD curve
ds.plot_macd(stock_data, stock_ticker, start_date, end_date)
# Do linear regression
lr_target_date, lr_X, lr_Y = lr.linear_regression_dataprep(stock_data)
lr.linear_regression(stock_data, stock_ticker, lr_target_date, lr_X, lr_Y)