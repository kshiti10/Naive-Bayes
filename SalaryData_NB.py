# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:05:57 2024

@author: Kshitija
Problem Statement:
1.) Prepare a classification model using the Naive Bayes algorithm for the salary dataset. Train and test datasets are given separately. Use both for model building.   
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
 
#####Loading data
email_train=pd.read_csv("C:/DataSet/SalaryData_Train.csv",encoding="ISO-8859-1")
email_test=pd.read_csv("C:/DataSet/SalaryData_Test.csv" ,encoding="ISO-8859-1")
 
from sklearn.model_selection import train_test_split
train_test_split(email_train,test_size=0.2)
 
########creating matrix of token counts for entire text documents####

emails_bow=CountVectorizer().fit(email_test.Salary) 
all_emails_matrix=emails_bow.transform(email_test.Salary)

####For training clients 

train_emails_matrix=emails_bow.transform(email_train.education)

###for testing clients
test_emails_matrix=emails_bow.transform(email_test.education)

#####Learning Term weightaging and normaling on entire clients salary
tfidf_transformer=TfidfTransformer().fit(all_emails_matrix)

######preparing TFIDF for train mails
train_tfidf=tfidf_transformer.transform(train_emails_matrix)

####preparing TFIDF for test mails
test_tfidf=tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape

######Now let us apply this to the Naive Bayes therorem

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,email_train.workclass)

######Evalution on test data

test_pred_m= classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.workclass) 
accuracy_test_m    
  




