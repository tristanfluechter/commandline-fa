import data_importer
import descriptive_stats as ds
import linear_regression as lr

stock_data, stock_ticker, start_date, end_date = data_importer.get_yahoo_data()

ds.show_stock_price(stock_ticker)
ds.describe_stock_data(stock_data, stock_ticker)
ds.plot_trendline(stock_data, stock_ticker, start_date, end_date)
ds.plot_simple_ma(stock_data, stock_ticker, start_date, end_date)
ds.plot_weighted_ma(stock_data, stock_ticker, start_date, end_date)
ds.plot_macd(stock_data, stock_ticker, start_date, end_date)

lr.linear_regression(stock_data)

