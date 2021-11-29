"""
This module provides a sentiment analysis for a given stock ticker
based on current Google News headlines.

Credit goes to https://www.analyticsvidhya.com/blog/2021/05/stock-price-movement-based-on-news-headline/
All customizing has been done by the authors of this program
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import regex as re
import os


def train_rf_model():
    """
    Trains a randomforest decision tree with a dataset including stock news and the movement of the stock 
    market the next day. Dataset can be found at:  https://github.com/ronylpatil/Stock-Sentiment-Analysis
    
    This model should only be used if a new model is to be created. The Random Forest Classifier
    has been pre-trained for quickness of use - therefore, train_rf_model() is not used
    in the current script.
    
    Returns: countvector, randomclassifier
    """
    
    cwd = os.getcwd()
    
    # Import base dataset. Labels: 0 == Stock going down, 1 == Stock going up.
    df = pd.read_csv(cwd + "/" "Stock News Training Dataset.csv", encoding = "ISO-8859-1")

    # Divide dataset into train and test
    train = df[df['Date'] < '20150101']
    test = df[df['Date'] > '20141231']

    print("Successfully imported news dataset!")

    # Remove special characters
    data = train.iloc[:, 2:27] # to only get the text columns
    data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

    # Create new columns for the 25 data columns
    list1 = [i for i in range(25)]
    new_Index=[str(i) for i in list1]
    data.columns = new_Index

    # convert everything to lowercase
    for index in new_Index:
        data[index] = data[index].str.lower()

    # create list to include all headlines into a single row
    headlines = []
    for row in range(0,len(data.index)):
        headlines.append(" ".join(str(x) for x in data.iloc[row, 0:25]))

    # implement BAG OF WORDS ML model
    countvector = CountVectorizer(ngram_range=(2,2))
    traindataset = countvector.fit_transform(headlines)
    vocab = countvector.vocabulary_

    # Save the countvector
    pickle.dump(countvector, open("vector.pickel", "wb"))

    # implement Random Forest Classifier
    randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
    randomclassifier.fit(traindataset, train['Label'])

    # Predict for the Test Dataset
    test_transform = []
    for row in range(0,len(test.index)):
        test_transform.append(" ".join(str(x) for x in test.iloc[row, 2:27]))

    test_dataset = countvector.transform(test_transform)
    predictions = randomclassifier.predict(test_dataset) # will predict a 0 or 1 outcome

    # how did the program do?
    score = accuracy_score(test['Label'], predictions) # tests score
    print(f"Score of random forest classifier: {score}")
    report = classification_report(test['Label'], predictions)
    print("Report of random classifier: ")
    print(report)

    # Save classifier
    filename = 'randomforest_sentiment_classifier.sav'
    pickle.dump(randomclassifier, open(filename, 'wb'))
    
    return countvector, randomclassifier

def rf_predict(stock_news):
    """
    Creates a binary prediction of whether or not the stock might be
    expected to increase (1) or decrease (0) in value on the following day.
    """
    
    # Load pre-trained countvector
    countvector = pickle.load(open("data/vector.pickel", "rb"))
    randomclassifier = pickle.load(open("data/randomforest_sentiment_classifier.sav", "rb"))
    
    # Get headlines from JSON object
    stock_news_headlines = []
    
    # Iterate through headlines
    for headline in range(len(stock_news)):
        stock_news_headlines.append(stock_news[headline]["title"])

    stock_news_headlines_temp = "".join(stock_news_headlines) # headlines are now strings

    stock_news_headlines = re.sub('[^A-Za-z0-9 ]+','',stock_news_headlines_temp)

    pred_headlines = ["".join(stock_news_headlines)] # join the list together as a string, then turn it into a list again to remove commas

    dataset_to_predict = countvector.transform(pred_headlines)

    predictions = randomclassifier.predict(dataset_to_predict)
    
    print(f"Analyzed current top 25 Google News headlines.")
    print("The trained Random Forest Classifier has calculated the headline sentiment.")

    if predictions == 1:
        print("Positive sentiment detected - buy stock.")
        rf_pred = "Positive Sentiment"
    else:
        print("Negative sentiment detected - sell stock.")
        rf_pred = "Negative Sentiment"
    
    return rf_pred