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
    # Create GNews object with restrictions: English, US-based, last 7 days, 5 results
    google_news = GNews(language='en', country='US', period='7d', max_results=5)
    stock_news = google_news.get_news(ticker + " Stock")
    
    # Create empty headlines list
    stock_news_headlines = []
    
    # Iterate through headlines
    for headline in range(len(stock_news)):
        stock_news_headlines.append(stock_news[headline]["title"])

    # Define headlines variables
    headline1 = stock_news_headlines[0]
    headline2 = stock_news_headlines[1]
    headline3 = stock_news_headlines[2]
    headline4 = stock_news_headlines[3]
    headline5 = stock_news_headlines[4]
    
    print(f"These are the top 5 Google News headlines for {ticker}.")
    print(headline1, "\n",headline2,"\n", headline3,"\n",headline4,"\n", headline5)
    return headline1, headline2, headline3, headline4, headline5

def main():
    get_headlines()
    
if __name__ == "__main__":
    main()