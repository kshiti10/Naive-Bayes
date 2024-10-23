# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:06:53 2024

@author: Kshitija
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Loading Data
DisasterTweets = pd.read_csv("C:/datasets/disaster_tweets_NB.csv", encoding = "ISO-8859-1")

# Cleaning of Data
import re

def cleaning_text(i):
    w = []
    i = re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word) > 3:
            w.append(word)
    return (" ".join(w))
# Testing above function with some test text
cleaning_text("Hope your are having good week.just checking")
cleaning_text("hope i can understand your feelings 123121.123.hi how are you?")
cleaning_text("hi how are you, I am sad")
DisasterTweets.text = DisasterTweets.text.apply(cleaning_text)
DisasterTweets = DisasterTweets.loc[DisasterTweets.text != "",:]
from sklearn.model_selection import train_test_split
DisasterTweets_train, DisasterTweets_test = train_test_split(DisasterTweets, test_size = 0.2)
# creating matrix of token counts for entire text document

def split_into_words(i):
    return[word for word in i.split(" ")]
DisasterTweets_bow = CountVectorizer(analyzer = split_into_words).fit(DisasterTweets.text)
all_DisasterTweets_matrix = DisasterTweets_bow.transform(DisasterTweets.text)
# for traing message
train_DisasterTweets_matrix = DisasterTweets_bow.transform(DisasterTweets.text)
# for testing message
test_DisasterTweets_matrix = DisasterTweets_bow.transform(DisasterTweets.text)

# learning Term weightaging and normaling on entire emails

tfidf_transformer = TfidfTransformer().fit(all_DisasterTweets_matrix)

# Preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_DisasterTweets_matrix)
# preparing TFIDF for test mails
test_tfidf = tfidf_transformer.transform(test_DisasterTweets_matrix)
test_tfidf.shape

# Naive Bayes Implementation

from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_tfidf, DisasterTweets_train.target)
# evaluation on test data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == DisasterTweets_test.target)
accuracy_test_m
