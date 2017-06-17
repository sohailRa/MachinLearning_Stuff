# CREDITS: SIRAJ RAVAL

import tweepy
from textblob import textblob

# Authentication
consumer_key= 'CONSUMER_KEY_HERE'
consumer_secret= 'CONSUMER_SECRET_HERE'

access_token='ACCESS_TOKEN_HERE'
access_token_secret='ACCESS_TOKEN_SECRET_HERE'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Retrieve Tweets
search_tweets = api.search('Trump')

for tweet in search_tweets:
	print(tweet.text)
	# Perform Sentiment Analysis on Tweets
	analysis = TextBlob(tweet.text)
	print(analysis.sentiment)
	print("")

