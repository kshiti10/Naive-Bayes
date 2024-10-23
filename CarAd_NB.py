# -- coding: utf-8 --
"""
Created on Mon Jan 29 21:16:19 2024

@author: Kshitija
This dataset contains information of users in a social network. This social 
network has several business clients which can post ads on it. One of the clients
 has a car company which has just launched a luxury SUV for a ridiculous price.
 Build a Bernoulli Naïve Bayes model using this dataset and classify which of the 
 users of the social network are going to purchase this luxury SUV. 1 implies that
 there was a purchase and 0 implies there wasn’t a purchase.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
 
#####Loading data
car_data=pd.read_csv("C:/DataSet/NB_Car_Ad.csv",encoding="ISO-8859-1")

#########cleaning of data
import re
def cleaning_text(i):
    w=[]
    i=re.sub("[0,1]+"," ",i).lower()
    for word in i.split(" "):
        if(len(word)>3):
            w.append(word)
    return (" ".join(w))

##############Testing above functions with some test text
cleaning_text("Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all")
cleaning_text("#raining #flooding #Florida #TampaBay #Tampa 18 or 19 days. I've lost count ")
cleaning_text("Hii,How are you I am  Sad")
 
car_data.text=car_data.Purchased.apply(cleaning_text)   
car_data=car_data.loc[car_data.text!="",:] 
  
from sklearn.model_selection import train_test_split
car_train,car_test=train_test_split(car_data,test_size=0.2)

########creating matrix of token counts for entire text documents####

def split_into_words(i):
    return[word for word in i.split(" ")]


car_bow=CountVectorizer(analyzer=split_into_words).fit(car_data.EstimaredSalary)
all_car_matrix=car_bow.transform(car_data.EstimaredSalary)

####For training messages

train_car_matrix=car_bow.transform(car_train.EstimaredSalary)

###for testing messages
test_car_matrix=car_bow.transform(car_test.EstimaredSalary)

#####Learning Term weightaging and normaling 
tfidf_transformer=TfidfTransformer().fit(all_car_matrix)

######preparing TFIDF 
train_tfidf=tfidf_transformer.transform(train_car_matrix)

####preparing TFIDF 
test_tfidf=tfidf_transformer.transform(test_car_matrix)
test_tfidf.shape

######Now let us apply this to the Naive Bayes therorem--

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,car_train.Purchased)

######Evalution on test data

test_pred_m= classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==car_test.Purchased) 
accuracy_test_m