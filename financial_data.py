"""
This module scrapes financial websites to obtain and output
financial data such as analyst predictions and key financial KPI.
It uses different websites to obtain this information since some website don't make the
presented values easily available.
"""

# Import relevant libraries
# html parser
from bs4 import BeautifulSoup
# web scraper
import requests
# regex to search for strings on sites
import re

def scrape_analyst_predictions(ticker):
    """
    This program uses beautiful soup to parse the CNN money page
    and returns the analyst predictions for a given stock ticker
    """
    # Get the CNN Money Website for a given stock ticker
    URL = f"https://money.cnn.com/quote/forecast/forecast.html?symb={ticker}"
    page = requests.get(URL)

    # Parse the website with beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")

    # The relevant website information has been isolated with "inspect element"
    # Find the Analysts Offering string in the parsed webpage
    find_string = soup.body.findAll(text=re.compile('analysts offering'), limit=1) # returns a list
    
    # Extract the string form find_string
    analyst_predictions_text = find_string[0]
    
    # Search and extract the median price prediction, highest and lowest price prediction.
    analyst_predictions = re.findall("\d+\.\d+", analyst_predictions_text)
    
    # Extract price information
    median_price = analyst_predictions[0]
    high_price = analyst_predictions[1]
    low_price = analyst_predictions[2]

    # Print out price information
    print(f"Analyst predictions for stock ticker {ticker}: ")
    print(f"Median price target according to CNN analysts is USD {median_price}.")
    print(f"Upper-end price target is USD {high_price}.")
    print(f"Lower-end price target is USD {low_price}.")    
    
def scrape_financial_kpi(ticker):
    """
    This program uses beautiful soup to parse the Marketwatch
    website to retrieve key financial KPI.
    """
    
    # Get the Market Watch website for a given stock ticker
    URL = f"https://www.marketwatch.com/investing/stock/{ticker}/financials"
    page = requests.get(URL)

    # Parse the website with beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Find the Analysts Offering string in the parsed webpage
    # Intraday change is either stored in a "positive" or "negative" table cell
    span_change_intraday = soup.body.find_all('span', {'class':"change--point--q"}) 
    span_percent_change_intraday = soup.body.find_all('span', {'class':"change--percent--q"}) 
  
    # Find absolute change and percent change in obtained data
    abs_change = re.findall("[-]?\d+\.\d+", str(span_change_intraday[0]))[1] # first value in table is abs change
    percent_change = re.findall("[-]?\d+\.\d+%", str(span_percent_change_intraday[0]))[1] # second value in table is % change
    
    # Print out intraday change information
    print(f"Intraday change and trading volume of {ticker}: ")
    print(f"Intraday stock value change: USD {abs_change}.")
    print(f"Intraday stock value change: {percent_change}%")
    
    # Find current trading volume
    span_trading_volume = soup.body.find_all('span', {'class':"primary"})
    trading_volume = re.findall("\d+\.\d+M", str(span_trading_volume[1]))[0]
    
    print(f"Trading volume: {trading_volume}")
    
def scrape_company_kpi(ticker):
    """
    This program scrapes Yahoo Finance to obtain further KPI that influence the decision
    to buy a certain stock.
    """
    # Get the Market Watch website for a given stock ticker
    URL = f"https://finance.yahoo.com/quote/{ticker}"
    page = requests.get(URL)

    # Parse the website with beautifulsoup
    soup = BeautifulSoup(page.content, "html.parser")
    
    eps = soup.body.find_all("td", {'data-test':"PE_RATIO-value"})
    actual_eps = re.findall("\d+\.\d+", str(eps[0]))[1]
    actual_eps 
     
    
    
def main():
    scrape_analyst_predictions("AAPL")
    scrape_financial_kpi("AAPL")

if __name__ == "__main__":
    main()