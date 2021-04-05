import pymongo
from pymongo import MongoClient

import json
import twitter
from pprint import pprint
import tweepy
import matplotlib.pyplot as plt
from textblob import TextBlob
import time
import pandas as pd
import numpy as np
import re
import string
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# # ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

CONSUMER_KEY = "8AO6OU5ubyi4XO47b1C7Sjdlz"
CONSUMER_SECRET ="FS1usPrfPolvjLXbwGka5N8TWkOZhUsdxGmmTwuO016koesUSt"

# After the step above, you will be redirected to your app's page.
# Create an access token under the the "Your access token" section
OAUTH_TOKEN ="1151573806680592384-OUFeUtpsRFZM6jQxl1AG99NEjlY0Kt"
OAUTH_TOKEN_SECRET ="KKHmkHkDGVaDof8XK4fKKI52DmNl4vZlaXnx85WRfd4Lr"

# dir(tweepy)
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

# auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter_api = tweepy.API(auth,)

client = MongoClient()
db = client.tweet_db
tweet_collection = db.tweet_collection
tweet_collection.create_index([("id", pymongo.ASCENDING)], unique=True)

####
# count=10

# search_results= twitter_api.search('India', count=count )
count = 50
q = "India"
search_results = twitter_api.search(count=count, q=q)
# pprint(search_results['search_metadata'])


# statuses = search_results["statuses"]

# since_id_new = statuses[-1]['id']

for statues in search_results:
    try:
        tweet_collection.insert_many(statues)
    except:
        pass

tweet_cursor = tweet_collection.find()
print(tweet_cursor.count())
user_cursor = tweet_collection.distinct("user.id")
print(len(user_cursor))

for document in tweet_cursor:
    try:
        pass
        # print('-----')
        # print('name:-', document["user"]["name"])
        # print('text:-', document["text"])
        # print('Created Date:-', document["created_at"])
    except:
        print("Error in Encoding")
        pass


pos = 0
neg = 0
neu = 0

# printing line by line
for tweet in search_results:
    # print(tweet.text)
    analysis = TextBlob(tweet.text)  # here it will apply NLP\
    print(analysis.sentiment)
    # now checking polarity only
    if analysis.sentiment.polarity > 0:
        print("positive")
        pos = pos + 1
    elif analysis.sentiment.polarity == 0:
        print("Neutral")
        neu = neu + 1
    else:
        print("Negative")
        neg = neg + 1

# ploting graphs
plt.xlabel("tags")
plt.ylabel("polarity")
plt.bar(['pos','neg','neu'],[pos,neg,neu])
plt.pie([pos, neg, neu], labels=['pos', 'neg', 'neu'], autopct="%5.5f%%")
plt.show()
####

# stop_words = set(stopwords.words('english'))
#
# def load_dataset(filename, cols):
#     dataset = pd.read_csv(search_results, encoding='latin-1')
#     dataset.columns = cols
#     return dataset
#
# def remove_unwanted_cols(dataset, cols):
#      for col in cols:
#          del dataset[col]
#      return dataset
# #
# #
# def preprocess_tweet_text(tweet):
#     tweet.lower()
#      # Remove urls
#     tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
#      # Remove user @ references and '#' from tweet
#     tweet = re.sub(r'\@\w+|\#', '', tweet)
#      # Remove punctuations
#     tweet = tweet.translate(str.maketrans('', '', string.punctuation))
#
# #     # Remove stopwords
#     tweet_tokens = word_tokenize(tweet)
#     filtered_words = [w for w in tweet_tokens if not w in stop_words]
#
#     return " ".join(filtered_words)
#
# # ps = PorterStemmer()
# #     # stemmed_words = [ps.stem(w) for w in filtered_words]
# #     # lemmatizer = WordNetLemmatizer()
# #     # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
# #
# def get_feature_vector(train_fit):
#     vector = TfidfVectorizer(sublinear_tf=True)
#     vector.fit(train_fit)
#     return vector
# #
# #
# def int_to_string(sentiment):
#      if sentiment == 0:
#          return "Negative"
#      elif sentiment == 2:
#          return "Neutral"
#      else:
#          return "Positive"
# #

# dataset = load_dataset(search_results, ['target', 't_id', 'created_at', 'query', 'user', 'text'])

# dataset = load_dataset("db", ['target', 't_id', 'created_at', 'query', 'user', 'text'])
# # # Remove unwanted columns from dataset
# n_dataset = remove_unwanted_cols(dataset, ['t_id', 'created_at', 'query', 'user'])
# # #Preprocess data
# dataset.text = dataset['text'].apply(preprocess_tweet_text)
# # # Split dataset into Train, Test
# #
# # # Same tf vector will be used for Testing sentiments on unseen trending data
# tf_vector = get_feature_vector(np.array(tweet_cursor.iloc[:, 1]).ravel())
# X = tf_vector.transform(np.array(dataset.iloc[:, 1]).ravel())
# y = np.array(dataset.iloc[:, 0]).ravel()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)


# # # Training Naive Bayes model
# NB_model = MultinomialNB()
# NB_model.fit(X_train, y_train)
# y_predict_nb = NB_model.predict(X_test)
# print(accuracy_score(y_test, y_predict_nb))
# #
# # # Training Logistics Regression model
# LR_model = LogisticRegression(solver='lbfgs')
# LR_model.fit(X_train, y_train)
# y_predict_lr = LR_model.predict(X_test)
# print(accuracy_score(y_test, y_predict_lr))

#
#
#
# test_file_name = "trending_tweets/08-04-2020-1586291553-tweets.csv"
# test_ds = load_dataset(test_file_name, ["t_id", "hashtag", "created_at", "user", "text"])
# test_ds = remove_unwanted_cols(test_ds, ["t_id", "created_at", "user"])
#
# # # Creating text feature
# test_ds.text = test_ds["text"].apply(preprocess_tweet_text)
# test_feature = tf_vector.transform(np.array(test_ds.iloc[:, 1]).ravel())
# #
# # # Using Logistic Regression model for prediction
# test_prediction_lr = LR_model.predict(test_feature)
# #
# # # Averaging out the hashtags result
# test_result_ds = pd.DataFrame({'hashtag': test_ds.hashtag, 'prediction':test_prediction_lr})
# test_result = test_result_ds.groupby(['hashtag']).max().reset_index()
# test_result.columns = ['heashtag', 'predictions']
# test_result.predictions = test_result['predictions'].apply(int_to_string)
# #
# print(test_result)

