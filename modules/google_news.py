""" 
This module can scrape Google News for headlines regarding a specific stock.
This is used both for descriptive analytics and for the optional sentiment analysis.
"""

# import relevant libraries
from gnews import GNews
import pandas as pd

def get_headlines(ticker):
    """
    This program retrieves Goole News headlines based on the ticker.
    """
    # Create GNews object with restrictions: English, US-based, last 7 days, 25 results
    google_news = GNews(language='en', country='US', period='7d', max_results=25)
    stock_news = google_news.get_news(ticker + " Stock")
    
    return stock_news

def print_headlines(stock_news, stock_ticker):
    
    # Create empty headlines list
    stock_news_headlines = []
    
    # Iterate through headlines to get top 5 headlines
    for headline in range(5):
        stock_news_headlines.append(stock_news[headline]["title"])

    # Define headlines variables
    headline1 = stock_news_headlines[0]
    headline2 = stock_news_headlines[1]
    headline3 = stock_news_headlines[2]
    headline4 = stock_news_headlines[3]
    headline5 = stock_news_headlines[4]
    
    # Write headlines
    print(f"These are the top 5 Google News headlines for {stock_ticker}:\n")
    
    for i in range(5):
        print(stock_news_headlines[i])